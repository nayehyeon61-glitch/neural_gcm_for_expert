"""Initialize, train and evaluate using official NeuralGCM pressure-level I/O."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import jax
import numpy as np
from neuralgcm import PressureLevelModel


def _json(path):
    return json.loads(Path(path).read_text()) if path else None


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def initialize(args):
    from neuralgcm_expert.model import load_checkpoint, from_checkpoint, save_checkpoint
    from neuralgcm_expert.data import open_dataset
    checkpoint = load_checkpoint(args.checkpoint)
    reference = PressureLevelModel.from_checkpoint(checkpoint)
    ds = open_dataset(args.data)
    sample = ds.isel(time=0) if 'time' in ds.dims else ds
    bindings = []
    if args.no_residual:
        if args.mode != 'experts':
            raise ValueError('--no-residual applies only to experts')
        bindings.append('DecomposedNodalMapping.enable_residual = False')
    if args.remat_towers:
        if args.mode == 'baseline':
            raise ValueError('--remat-towers does not apply to the untouched baseline')
        # Recompute column towers in the backward pass: identical values, but five
        # expert towers otherwise store every column activation (~117 -> ~17 GiB).
        bindings.extend(f'{scope}/ColumnTower.checkpoint_tower = True'
                        for scope in ('encode', 'process', 'decode'))
    model = from_checkpoint(checkpoint, mode=args.mode,
                            sample_inputs=reference.inputs_from_xarray(sample),
                            sample_forcings=reference.forcings_from_xarray(sample),
                            rng=jax.random.PRNGKey(args.seed), gin_bindings=bindings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, output, checkpoint)
    report = getattr(model, 'expert_metadata', {})
    _write_json(output.with_suffix('.json'), report)
    transfer = report.get('parameter_transfer', {})
    print(json.dumps({'mode': args.mode, 'checkpoint': str(output),
                      'total_parameters': report.get('total_parameter_count'),
                      'transferred_leaves': len(transfer.get('transferred', [])),
                      'initialized_leaves': len(transfer.get('initialized', [])),
                      'report': str(output.with_suffix('.json'))}, indent=2))


def evaluate_command(args):
    from neuralgcm_expert.model import load_checkpoint, restore_checkpoint
    from neuralgcm_expert import mappings  # register custom Gin classes
    from neuralgcm_expert.evaluation import evaluate, evaluate_many
    from neuralgcm_expert.data import open_dataset
    del mappings
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_checkpoint(checkpoint)
    ds = open_dataset(args.data)
    climatology = open_dataset(args.climatology) if args.climatology else None
    evaluator = evaluate_many if args.all_starts else evaluate
    extra = {'stride': args.stride, 'store_predictions': args.store_predictions} if args.all_starts else {}
    result = evaluator(model, ds, forecast_steps=args.steps,
                       output_interval=args.interval, climatology=climatology,
                       forcing_mode=args.forcing_mode, **extra)
    result.save(args.output)
    print(f'Evaluation saved: {args.output}')


def train(args):
    from neuralgcm_expert.model import load_checkpoint, save_checkpoint, restore_checkpoint
    from neuralgcm_expert import mappings
    from neuralgcm_expert.data import open_dataset, make_training_window, _require_time, output_timedelta
    from neuralgcm_expert.evaluation import (
        complete_forecast_starts, evaluate_many, normalized_forecast_mse, quadrature_weights)
    from neuralgcm_expert.training import (
        LossConfig, estimate_scales, make_train_step, prepare_native_targets,
        window_batch, with_params, trainable_mask)
    del mappings
    if args.epochs < 1 or args.stride < 1 or args.validation_stride < 1:
        raise ValueError('epochs, stride and validation-stride must be positive')
    validation_steps = args.steps if args.validation_steps is None else args.validation_steps
    if args.steps < 1 or validation_steps < 1:
        raise ValueError('steps and validation-steps must be positive')
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_checkpoint(checkpoint)
    ds = open_dataset(args.train_data)
    valid = open_dataset(args.validation_data) if args.validation_data else None
    times = _require_time(ds)
    interval = output_timedelta(model, args.interval)
    horizon = max(args.steps, 2 if (args.tendency_weight or args.acceleration_weight) else 1)
    indices = complete_forecast_starts(model, ds, horizon, interval, stride=args.stride)
    validation_indices = None
    if valid is not None:
        valid_times = _require_time(valid)
        # All overlapping time ranges are rejected, not merely identical samples.
        if not (times[-1] < valid_times[0] or valid_times[-1] < times[0]):
            raise ValueError('Training and validation time ranges overlap')
        validation_indices = complete_forecast_starts(
            model, valid, validation_steps, interval, stride=args.validation_stride)
    variable_scales = _json(args.scales) or estimate_scales(ds, model.input_variables)
    if set(variable_scales) != set(model.input_variables):
        raise ValueError('Scale file must contain exactly model.input_variables')
    if any(np.ndim(value) != 0 or not np.isfinite(value) or value <= 0
           for value in variable_scales.values()):
        raise ValueError('Scale file must contain finite positive scalar values')
    native_scales = _json(args.native_scales)
    config = LossConfig(tendency_weight=args.tendency_weight,
                        acceleration_weight=args.acceleration_weight,
                        residual_parameter_weight=args.residual_weight)
    _, opt_state, update = make_train_step(
        model, forecast_steps=args.steps, output_interval=args.interval,
        variable_scales=variable_scales,
        spatial_weights=quadrature_weights(model.data_coords.horizontal),
        loss_config=config, learning_rate=args.learning_rate, scope=args.scope,
        native_scales=native_scales)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/'scales.json', variable_scales)
    _write_json(output/'run.json', {k: v for k, v in vars(args).items() if k != 'func'})
    rng = np.random.default_rng(args.seed)
    params = model.params
    mask = trainable_mask(params, args.scope)
    trainable_count = sum(value.size for value, enabled in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(mask)) if enabled)
    native_reference = model  # fixed encoder targets, even if --scope all is used
    best_validation = float('inf')
    best_epoch = None
    with (output/'training.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            order = rng.permutation(indices)
            for index in order:
                window = make_training_window(model, ds, args.steps,
                                              output_interval=args.interval,
                                              start_index=int(index),
                                              forcing_mode=args.forcing_mode)
                batch = window_batch(window)
                if args.tendency_weight or args.acceleration_weight:
                    triplet_times = times[index] + np.arange(3)*interval
                    triplet = ds.sel(time=triplet_times)
                    native = prepare_native_targets(native_reference, triplet, args.interval,
                                                     native_scales=native_scales)
                    native.pop('scales')  # static scales are closed over in the loss
                    batch['native'] = native
                params, opt_state, metrics = update(params, opt_state, batch)
                metrics = {key: float(value) for key, value in jax.device_get(metrics).items()}
                if not metrics['accepted']:
                    raise FloatingPointError('Nonfinite loss/gradient; update rejected; inspect scales and horizon')
                record = {'epoch': epoch+1, 'start_index': int(index), **metrics}
                log.write(json.dumps(record, allow_nan=False)+'\n')
                log.flush()
                print(json.dumps(record), flush=True)
            model = with_params(model, params)
            model.expert_metadata.update({
                'training_completed_epochs': epoch+1,
                'training_scope': args.scope,
                'trainable_parameter_count': trainable_count,
                'training_data_start': str(times[0]), 'training_data_end': str(times[-1]),
                'forecast_steps': args.steps, 'forecast_interval': args.interval,
                'requires_training': False,
                'validation_status': 'evaluated separately' if valid is not None else 'not evaluated',
                'note': 'Training executed; forecast skill and long-term stability are not certified.'})
            save_checkpoint(model, output/'last.pkl', checkpoint)
            if valid is not None:
                # A freshly initialized architecture can blow up in the validation
                # rollout; score that epoch as +inf (never selected) and keep training.
                try:
                    result = evaluate_many(
                        model, valid, forecast_steps=validation_steps,
                        output_interval=args.interval, forcing_mode=args.forcing_mode,
                        stride=args.validation_stride, store_predictions=False)
                except ValueError as exc:
                    if 'non-finite' not in str(exc):
                        raise
                    result, score, failure = None, float('inf'), str(exc)
                else:
                    result.save(output/f'validation-epoch-{epoch+1}')
                    # Pool initialization MSE before normalization; the same fixed
                    # training scales and equal lead/level/variable averaging apply.
                    score = normalized_forecast_mse(
                        result.metrics, variable_scales,
                        expected_initialization_count=len(validation_indices))
                    failure = None
                if score < best_validation:
                    best_validation = score
                    best_epoch = epoch+1
                    save_checkpoint(model, output/'best.pkl', checkpoint)
                selection = {
                    'epoch': epoch+1,
                    'current_normalized_mse': score if np.isfinite(score) else None,
                    'current_failure': failure,
                    'best_epoch': best_epoch,
                    'best_normalized_mse': best_validation if np.isfinite(best_validation) else None,
                    'initialization_count': len(validation_indices),
                    'initialization_stride': args.validation_stride,
                    'forecast_steps': validation_steps,
                    'output_interval_hours': float(interval / np.timedelta64(1, 'h')),
                    'forecast_horizon_hours': float(validation_steps*interval / np.timedelta64(1, 'h')),
                    'initialization_start': (result.metadata['initialization_start'] if result is not None
                                             else str(valid_times[validation_indices[0]])),
                    'initialization_end': (result.metadata['initialization_end'] if result is not None
                                           else str(valid_times[validation_indices[-1]])),
                    'forcing_mode': args.forcing_mode,
                    'selection': 'all complete validation initializations at configured stride; all requested leads',
                    'protocol': 'pool initialization MSE; divide by training scale squared; mean over leads, levels, then variables',
                    'scale_source': 'scales.json (training-only normalization)',
                    'test_data_used': False}
                _write_json(output/'validation-selection.json', selection)
                print(json.dumps({'validation': selection}), flush=True)
    print(f'Training checkpoint saved: {output / "last.pkl"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='create a deterministic experiment from an official checkpoint')
    init.add_argument('--checkpoint', required=True)
    init.add_argument('--data', required=True, help='prepared official-grid ERA5 NetCDF or Zarr')
    init.add_argument('--output', required=True)
    init.add_argument('--mode', choices=['baseline', 'memoryless', 'experts'], default='experts')
    init.add_argument('--no-residual', action='store_true')
    init.add_argument('--remat-towers', action='store_true',
                      help='rematerialize column towers (memory only; same values)')
    init.add_argument('--seed', type=int, default=0)
    init.set_defaults(func=initialize)
    evaluation = commands.add_parser('evaluate', help='official pressure-level rollout and quadrature RMSE')
    evaluation.add_argument('--checkpoint', required=True)
    evaluation.add_argument('--data', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--steps', type=int, default=4)
    evaluation.add_argument('--interval', default='6h')
    evaluation.add_argument('--forcing-mode', choices=['persistence', 'dynamic'], default='persistence')
    evaluation.add_argument('--climatology')
    evaluation.add_argument('--all-starts', action='store_true', help='pool error statistics across all complete initializations')
    evaluation.add_argument('--stride', type=int, default=1)
    evaluation.add_argument('--store-predictions', action='store_true', help='retain every multi-initialization forecast in memory and NetCDF (can be large)')
    evaluation.set_defaults(func=evaluate_command)
    training = commands.add_parser('train', help='train through the official coupled solver')
    training.add_argument('--checkpoint', required=True)
    training.add_argument('--train-data', required=True)
    training.add_argument('--validation-data')
    training.add_argument('--validation-stride', type=int, default=1,
                          help='validation initialization stride in dataset samples (default: every complete start)')
    training.add_argument('--validation-steps', type=int,
                          help='checkpoint-selection horizon in output steps (default: --steps)')
    training.add_argument('--output', required=True)
    training.add_argument('--epochs', type=int, default=10)
    training.add_argument('--steps', type=int, default=4)
    training.add_argument('--interval', default='6h')
    training.add_argument('--stride', type=int, default=1)
    training.add_argument('--learning-rate', type=float, default=1e-4)
    training.add_argument('--scope', choices=['physics', 'all'], default='physics')
    training.add_argument('--scales', help='JSON: decoded variable name -> training normalization scale')
    training.add_argument('--native-scales', help='JSON: native field path -> positive nondimensional scale')
    training.add_argument('--tendency-weight', type=float, default=0.0)
    training.add_argument('--acceleration-weight', type=float, default=0.0)
    training.add_argument('--residual-weight', type=float, default=1e-6)
    training.add_argument('--forcing-mode', choices=['persistence', 'dynamic'], default='persistence')
    training.add_argument('--seed', type=int, default=0)
    training.set_defaults(func=train)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
