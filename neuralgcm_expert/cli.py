"""Initialize, train and evaluate using official NeuralGCM pressure-level I/O."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import jax
import numpy as np
import pandas as pd
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
    from neuralgcm_expert.data import open_dataset, make_training_window
    from neuralgcm_expert.evaluation import evaluate, quadrature_weights
    from neuralgcm_expert.training import (
        LossConfig, estimate_scales, make_train_step, prepare_native_targets,
        window_batch, with_params, trainable_mask)
    del mappings
    if args.epochs < 1 or args.stride < 1:
        raise ValueError('epochs and stride must be positive')
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_checkpoint(checkpoint)
    ds = open_dataset(args.train_data)
    valid = open_dataset(args.validation_data) if args.validation_data else None
    if valid is not None:
        # All overlapping time ranges are rejected, not merely identical samples.
        if not (ds.time.values[-1] < valid.time.values[0]
                or valid.time.values[-1] < ds.time.values[0]):
            raise ValueError('Training and validation time ranges overlap')
    variable_scales = _json(args.scales) or estimate_scales(ds, model.input_variables)
    if set(variable_scales) != set(model.input_variables):
        raise ValueError('Scale file must contain exactly model.input_variables')
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
    times = ds.time.values
    interval = np.timedelta64(pd.Timedelta(args.interval).value, 'ns')
    if interval <= np.timedelta64(0, 'ns'):
        raise ValueError('interval must be positive')
    horizon = max(args.steps, 2 if (args.tendency_weight or args.acceleration_weight) else 1)
    indices = [i for i in range(0, len(times), args.stride)
               if times[i] + horizon*interval <= times[-1]]
    if not indices:
        raise ValueError('No complete training windows for the requested forecast horizon')
    rng = np.random.default_rng(args.seed)
    params = model.params
    mask = trainable_mask(params, args.scope)
    trainable_count = sum(value.size for value, enabled in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(mask)) if enabled)
    native_reference = model  # fixed encoder targets, even if --scope all is used
    best_validation = float('inf')
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
                result = evaluate(model, valid, forecast_steps=args.steps,
                                  output_interval=args.interval, forcing_mode=args.forcing_mode)
                result.save(output/f'validation-epoch-{epoch+1}')
                # Checkpoint selection uses the same dimensionless rollout MSE,
                # avoiding averaging RMSE with incompatible physical units.
                window = make_training_window(model, valid, args.steps,
                                               output_interval=args.interval,
                                               forcing_mode=args.forcing_mode)
                from neuralgcm_expert.losses import weighted_mse
                pred = model.inputs_from_xarray(result.predictions)
                score = float(weighted_mse(
                    {k: pred[k] for k in variable_scales},
                    {k: window.targets[k] for k in variable_scales},
                    variable_scales=variable_scales,
                    spatial_weights=quadrature_weights(model.data_coords.horizontal)))
                if score < best_validation:
                    best_validation = score
                    save_checkpoint(model, output/'best.pkl', checkpoint)
                _write_json(output/'validation-selection.json', {
                    'best_normalized_mse': best_validation,
                    'selection': 'first validation initialization, all requested leads',
                    'test_data_used': False})
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
