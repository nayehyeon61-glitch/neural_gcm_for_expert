"""Stage a paired pretrained / fine-tuned / decomposed NeuralGCM experiment.

Inputs must already follow the official checkpoint grid and variable contract.
Training uses train+validation only; test is a separate command after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import jax
import numpy as np

from neuralgcm_expert.data import (
    _require_time, make_training_window, open_dataset, output_timedelta,
    validate_coordinates,
)
from neuralgcm_expert.model import restore_checkpoint


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def check_backend(allow_cpu):
    devices = jax.devices()
    if not allow_cpu and not any(device.platform == 'gpu' for device in devices):
        raise RuntimeError(
            'JAX sees no GPU. Install the pinned CUDA build and check jax.devices(). '
            '--allow-cpu is intended for a small smoke check only.')
    print('JAX devices:', devices, flush=True)


def inspect_data(path, model, steps, interval):
    """Check coordinates, uniform times, required variables and one full window.

    All later windows are checked by the trainer/evaluator as they are consumed.
    Nothing is silently regridded, renamed or replaced with a constant.
    """
    with open_dataset(path) as ds:
        times = _require_time(ds)
        validate_coordinates(ds, model)
        required = set(model.input_variables) | set(model.forcing_variables)
        missing = required - set(ds.data_vars)
        if missing:
            raise ValueError(f'{path}: missing required variables: {sorted(missing)}')
        if len(times) < 2:
            raise ValueError(f'{path}: at least two time samples are required')
        spacing = times[1] - times[0]
        if not np.all(np.diff(times) == spacing):
            raise ValueError(f'{path}: comparison workflow requires regular, gap-free times')
        output_step = output_timedelta(model, interval)
        if output_step < spacing or output_step % spacing != np.timedelta64(0, 'ns'):
            raise ValueError(f'{path}: output interval must be a multiple of data spacing')
        make_training_window(model, ds, steps, output_interval=interval)
        return {
            'path': str(Path(path).resolve()), 'start': str(times[0]),
            'end': str(times[-1]), 'samples': len(times),
            'sample_interval_hours': float(spacing / np.timedelta64(1, 'h')),
            'input_variables': list(model.input_variables),
            'forcing_variables': list(model.forcing_variables),
        }


def check_order(earlier, later):
    if np.datetime64(earlier['end']) >= np.datetime64(later['start']):
        raise ValueError('Require chronological, non-overlapping train < validation < test periods')


def run_cli(*arguments):
    command = [sys.executable, '-m', 'neuralgcm_expert.cli', *map(str, arguments)]
    print('+ ' + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def train_comparison(args):
    check_backend(args.allow_cpu)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; choose a new experiment directory')
    if any(value < 1 for value in (args.epochs, args.train_steps, args.evaluation_steps,
                                   args.train_stride, args.validation_stride, args.test_stride)):
        raise ValueError('Epochs, steps and strides must be positive')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    model = restore_checkpoint(args.checkpoint)
    train_info = inspect_data(args.train_data, model, args.train_steps, args.interval)
    valid_info = inspect_data(args.validation_data, model, args.evaluation_steps, args.interval)
    check_order(train_info, valid_info)
    if train_info['sample_interval_hours'] != valid_info['sample_interval_hours']:
        raise ValueError('Use the same time sampling in train and validation')
    modes = ['baseline', 'experts']
    if args.include_memoryless:
        modes.insert(1, 'memoryless')
    if args.scratch_width:
        # Physics from scratch: a single wide network vs the experts, matched in size.
        # The untouched baseline is initialized only as the pretrained test reference.
        if args.include_memoryless:
            raise ValueError('--scratch-width does not combine with --include-memoryless')
        modes = ['baseline', 'wide', 'experts']
    trained = [m for m in modes if not (args.scratch_width and m == 'baseline')]
    output.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    with args.checkpoint.open('rb') as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
    protocol = {
        'phase': 'training', 'source_checkpoint': str(args.checkpoint.resolve()),
        'source_checkpoint_sha256': checkpoint_sha256,
        'repository_commit': revision,
        'versions': {name: importlib.metadata.version(name)
                     for name in ('jax', 'jaxlib', 'neuralgcm', 'dinosaur')},
        'train': train_info, 'validation': valid_info, 'modes': modes,
        'trained_modes': trained, 'physics_from_scratch': bool(args.scratch_width),
        'wide_physics_width': args.scratch_width,
        'interval': args.interval, 'train_steps': args.train_steps,
        'evaluation_steps': args.evaluation_steps, 'epochs': args.epochs,
        'train_stride': args.train_stride, 'validation_stride': args.validation_stride,
        'test_stride': args.test_stride, 'learning_rate': args.learning_rate,
        'seed': args.seed, 'scope': 'physics', 'forcing_mode': 'persistence',
        'tendency_weight': 0.0, 'acceleration_weight': 0.0, 'residual_parameter_weight': 1e-6,
        'selection': 'pooled multi-initialization validation normalized forecast MSE',
        'tower_remat': 'all modes (memory only; identical values)',
        'rollout_remat': 'one saved state per output interval when train_steps > 1 (identical values)',
        'expert_output_init': 'zero (expert and residual output layers)',
        'expert_native_memory': 'kept (official memory encoder and memory features)',
        'warmup_steps': args.warmup_steps,
        'comparison_scope': (
            'learned physics from scratch (encoder/decoder official, frozen); single wide network '
            'vs experts at matched physics parameter count' if args.scratch_width else
            'transfer-learning experiment; not matched pretraining or parameter budgets'),
    }
    save_json(output / 'protocol.json', protocol)
    for mode in modes:
        extra = ['--zero-expert-output', '--keep-memory'] if mode == 'experts' else []
        if mode == 'wide':
            extra = ['--zero-expert-output', '--physics-width', args.scratch_width]
        if args.scratch_width and mode != 'baseline':
            extra.append('--reinit-physics')
        run_cli('init', '--mode', mode, '--checkpoint', args.checkpoint,
                '--data', args.train_data, '--seed', args.seed, '--remat-towers', *extra,
                '--output', output / 'checkpoints' / f'{mode}-init.pkl')
    # The original model is fine-tuned first, providing ONE train-derived scale
    # file that every architecture uses, including validation model selection.
    if getattr(args, 'init_only', False):
        print(f'Initialized {output}; train each arm with: resume --run {output} --mode <mode>',
              flush=True)
        return
    for mode in trained:
        train_arm(output, protocol, mode)
    protocol['phase'] = 'trained'
    save_json(output / 'protocol.json', protocol)
    print(f'Train and validation complete. Review {output}/protocol.json and each '
          'validation-selection.json before the separate test command.', flush=True)


def arm_label(mode):
    return 'baseline_finetuned' if mode == 'baseline' else mode


def trained_modes(protocol):
    return protocol.get('trained_modes', protocol['modes'])


def train_arm(output, protocol, mode, wait_for_scales=False):
    label = arm_label(mode)
    command = [
        'train', '--checkpoint', output / 'checkpoints' / f'{mode}-init.pkl',
        '--train-data', protocol['train']['path'],
        '--validation-data', protocol['validation']['path'],
        '--steps', protocol['train_steps'], '--interval', protocol['interval'],
        '--validation-steps', protocol['evaluation_steps'],
        '--validation-stride', protocol['validation_stride'],
        '--stride', protocol['train_stride'], '--epochs', protocol['epochs'],
        '--learning-rate', protocol['learning_rate'], '--seed', protocol['seed'],
        '--warmup-steps', protocol['warmup_steps'],
        '--scope', 'physics', '--forcing-mode', 'persistence',
        '--tendency-weight', 0, '--acceleration-weight', 0, '--residual-weight', 1e-6,
        '--output', output / label,
    ]
    # The first trained arm estimates ONE train-derived scale file that every
    # other arm uses, including validation model selection.
    first = trained_modes(protocol)[0]
    if mode != first:
        scales = output / arm_label(first) / 'scales.json'
        while wait_for_scales and not scales.is_file():  # arms run in parallel
            print(f'waiting for {scales}', flush=True)
            time.sleep(60)
        command.extend(['--scales', scales])
    run_cli(*command)


def arm_complete(output, protocol, label):
    selection = output / label / 'validation-selection.json'
    return (selection.is_file() and (output / label / 'best.pkl').is_file() and
            json.loads(selection.read_text())['epoch'] == protocol['epochs'])


def resume_comparison(args):
    """Finish the arms of an interrupted run with its frozen protocol and init checkpoints.

    Completed arms are kept; a partially trained arm is archived and retrained
    from its init checkpoint (training is not resumed mid-epoch).
    """
    check_backend(args.allow_cpu)
    output = args.run.resolve()
    protocol = json.loads((output / 'protocol.json').read_text())
    if protocol['phase'] != 'training' or (output / 'test').exists():
        raise ValueError('resume applies only to an interrupted, untested run')
    for mode in protocol['modes']:
        if not (output / 'checkpoints' / f'{mode}-init.pkl').is_file():
            raise FileNotFoundError(output / 'checkpoints' / f'{mode}-init.pkl')
    modes = trained_modes(protocol)
    only = getattr(args, 'mode', None)
    if only is not None:
        if only not in modes:
            raise ValueError(f'{only} is not a trained arm of this run: {modes}')
        modes = [only]
    for mode in modes:
        label = arm_label(mode)
        if arm_complete(output, protocol, label):
            print(f'{label}: complete, kept', flush=True)
            continue
        partial = output / label
        if partial.exists():
            archive = output / f'{label}.interrupted-{time.strftime("%Y%m%d-%H%M%S")}'
            partial.rename(archive)
            print(f'{label}: partial run archived to {archive.name}', flush=True)
        train_arm(output, protocol, mode, wait_for_scales=only is not None)
    # With arms on separate GPUs, the last arm to finish marks the run trained.
    if all(arm_complete(output, protocol, arm_label(m)) for m in trained_modes(protocol)):
        protocol = json.loads((output / 'protocol.json').read_text())
        protocol['phase'] = 'trained'
        save_json(output / 'protocol.json', protocol)


def concept_comparison(args):
    """Add the concept-bottleneck arm to an existing run, on the same protocol.

    The shared network is the official pretrained physics tower (its output
    channels are the temperature/momentum/moisture concept heads; no residual
    bypass). Training adds per-concept native ODE losses to the same rollout loss.
    """
    check_backend(args.allow_cpu)
    output = args.run.resolve()
    protocol = json.loads((output / 'protocol.json').read_text())
    # Extra named arms (e.g. a lambda ablation) may join after the test; they
    # select on validation as usual and are tested on their own with test-arm.
    name = args.name
    destination = output / name
    if destination.exists():
        raise FileExistsError(f'{destination} already exists')
    if name == 'concept' and (output / 'test').exists():
        raise ValueError('Test already ran; a new arm cannot join this run')
    record = output / f'{name}_protocol.json'
    scales = output / 'baseline_finetuned' / 'scales.json'
    if not scales.is_file():
        raise FileNotFoundError(scales)
    weights = {'temperature': args.ode_temperature_weight, 'momentum': args.ode_momentum_weight,
               'moisture': args.ode_moisture_weight}
    if not all(np.isfinite(w) and w > 0 for w in weights.values()):
        raise ValueError('concept ODE weights must be finite and positive')
    save_json(record, {
        'phase': 'training', 'name': name, 'base_protocol': protocol['repository_commit'],
        'init': 'checkpoints/baseline-init.pkl (official shared physics network)',
        'ode_weights': weights, 'ode_target': args.ode_target,
        'ode_target_definition': {
            'interval': 'per output interval j: P_k (x_hat_j - x*_0)/(j dt) vs P_k (x*_j - x*_0)/(j dt), '
                        'x_hat from the trained solver rollout, x* the frozen encoder on ERA5',
            'instantaneous': 'P_k native RHS(x*_1) vs P_k central difference'}[args.ode_target],
        'native_scales': 'RMS training native tendency of the same target (64 training triplets)'})
    command = [
        'train', '--checkpoint', output / 'checkpoints' / 'baseline-init.pkl',
        '--train-data', args.train_data, '--validation-data', args.validation_data,
        '--steps', protocol['train_steps'], '--interval', protocol['interval'],
        '--validation-steps', protocol['evaluation_steps'],
        '--validation-stride', protocol['validation_stride'],
        '--stride', protocol['train_stride'], '--epochs', protocol['epochs'],
        '--learning-rate', protocol['learning_rate'], '--seed', protocol['seed'],
        '--warmup-steps', protocol['warmup_steps'],
        '--scope', 'physics', '--forcing-mode', 'persistence',
        '--tendency-weight', 0, '--acceleration-weight', 0, '--residual-weight', 1e-6,
        '--ode-temperature-weight', weights['temperature'],
        '--ode-momentum-weight', weights['momentum'],
        '--ode-moisture-weight', weights['moisture'], '--ode-target', args.ode_target,
        '--scales', scales, '--output', destination]
    phase = 'failed'
    try:
        run_cli(*command)
        phase = 'trained'
    finally:
        save_json(record, {**json.loads(record.read_text()), 'phase': phase})


def test_comparison(args):
    check_backend(args.allow_cpu)
    output = args.run.resolve()
    protocol = json.loads((output / 'protocol.json').read_text())
    if protocol['phase'] != 'trained':
        raise ValueError('Test requires a completed training run with frozen validation selection')
    destination = output / 'test'
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; preserve existing test results')
    reference = output / 'checkpoints' / 'baseline-init.pkl'
    model = restore_checkpoint(reference)
    info = inspect_data(args.test_data, model, protocol['evaluation_steps'], protocol['interval'])
    check_order(protocol['validation'], info)
    if info['sample_interval_hours'] != protocol['train']['sample_interval_hours']:
        raise ValueError('Use the same time sampling in train, validation and test')
    choices = {'baseline_pretrained': reference}
    for mode in trained_modes(protocol):
        choices[arm_label(mode)] = output / arm_label(mode) / 'best.pkl'
    concept = output / 'concept_protocol.json'
    if concept.is_file():
        # A concept arm running in parallel joins the one-shot test once done.
        while json.loads(concept.read_text())['phase'] == 'training':
            print('waiting for the concept arm to finish training', flush=True)
            time.sleep(300)
        if json.loads(concept.read_text())['phase'] == 'trained':
            choices['concept'] = output / 'concept' / 'best.pkl'
    for path in choices.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.mkdir()
    save_json(destination / 'protocol.json', {**protocol, 'test': info})
    for label, checkpoint in choices.items():
        run_cli('evaluate', '--checkpoint', checkpoint, '--data', args.test_data,
                '--steps', protocol['evaluation_steps'], '--interval', protocol['interval'],
                '--all-starts', '--stride', protocol['test_stride'],
                '--forcing-mode', protocol['forcing_mode'], '--output', destination / label)
    comparison = Path(__file__).with_name('compare_metrics.py')
    candidates = [name for name in ('wide', 'experts', 'concept') if name in choices]
    for candidate in candidates:
        for baseline in choices:
            if baseline in candidates:
                continue
            subprocess.run([
                sys.executable, str(comparison), '--baseline', str(destination / baseline),
                '--candidate', str(destination / candidate),
                '--output', str(destination / f'{candidate}_vs_{baseline}.csv'),
            ], check=True)
    if 'wide' in choices:
        subprocess.run([
            sys.executable, str(comparison), '--baseline', str(destination / 'wide'),
            '--candidate', str(destination / 'experts'),
            '--output', str(destination / 'experts_vs_wide.csv'),
        ], check=True)
    if 'concept' in choices and 'experts' in choices:
        subprocess.run([
            sys.executable, str(comparison), '--baseline', str(destination / 'experts'),
            '--candidate', str(destination / 'concept'),
            '--output', str(destination / 'concept_vs_experts.csv'),
        ], check=True)
    protocol.update(phase='tested', test=info)
    save_json(output / 'protocol.json', protocol)
    print(f'Paired test results and improvement CSVs saved in {destination}', flush=True)


def test_arm(args):
    """Test one extra named arm against the arms already in the one-shot test."""
    check_backend(args.allow_cpu)
    output = args.run.resolve()
    protocol = json.loads((output / 'protocol.json').read_text())
    if protocol['phase'] != 'tested':
        raise ValueError('test-arm needs the main one-shot test to have run')
    if json.loads((output / f'{args.name}_protocol.json').read_text())['phase'] != 'trained':
        raise ValueError(f'{args.name} has not finished training')
    if args.test_data.resolve() != Path(protocol['test']['path']).resolve():
        raise ValueError('Use the same test data as the main test')
    tested = output / 'test'
    destination = tested / args.name
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; preserve existing test results')
    checkpoint = output / args.name / 'best.pkl'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    references = sorted(path.name for path in tested.iterdir()
                        if path.is_dir() and (path / 'metrics.nc').is_file())
    run_cli('evaluate', '--checkpoint', checkpoint, '--data', args.test_data,
            '--steps', protocol['evaluation_steps'], '--interval', protocol['interval'],
            '--all-starts', '--stride', protocol['test_stride'],
            '--forcing-mode', protocol['forcing_mode'], '--output', destination)
    comparison = Path(__file__).with_name('compare_metrics.py')
    for baseline in references:
        subprocess.run([
            sys.executable, str(comparison), '--baseline', str(tested / baseline),
            '--candidate', str(destination),
            '--output', str(tested / f'{args.name}_vs_{baseline}.csv'),
        ], check=True)
    print(f'{args.name} test results saved in {destination}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest='phase', required=True)
    train = phases.add_parser('train', help='initialize, train and select on validation')
    train.add_argument('--checkpoint', type=Path, required=True)
    train.add_argument('--train-data', type=Path, required=True)
    train.add_argument('--validation-data', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--interval', default='24h')
    train.add_argument('--train-steps', type=int, default=1)
    train.add_argument('--evaluation-steps', type=int, default=5)
    train.add_argument('--epochs', type=int, default=10)
    train.add_argument('--train-stride', type=int, default=1)
    train.add_argument('--validation-stride', type=int, default=7)
    train.add_argument('--test-stride', type=int, default=1)
    train.add_argument('--learning-rate', type=float, default=1e-4)
    train.add_argument('--warmup-steps', type=int, default=1000,
                       help='linear warmup updates, shared by every arm')
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--include-memoryless', action='store_true')
    train.add_argument('--init-only', action='store_true',
                       help='write protocol and init checkpoints; train arms with resume --mode')
    train.add_argument('--scratch-width', type=int,
                       help='physics from scratch: train a single physics network of this width '
                            'and the experts (both reinitialized); baseline is test reference only')
    train.add_argument('--allow-cpu', action='store_true')
    train.set_defaults(func=train_comparison)
    resume = phases.add_parser('resume', help='finish the arms of an interrupted run')
    resume.add_argument('--run', type=Path, required=True)
    resume.add_argument('--mode', help='train only this arm (arms can run on separate GPUs)')
    resume.add_argument('--allow-cpu', action='store_true')
    resume.set_defaults(func=resume_comparison)
    concept = phases.add_parser('concept', help='add the concept ODE-loss arm to a run')
    concept.add_argument('--run', type=Path, required=True)
    concept.add_argument('--train-data', type=Path, required=True)
    concept.add_argument('--validation-data', type=Path, required=True)
    for group in ('temperature', 'momentum', 'moisture'):
        concept.add_argument(f'--ode-{group}-weight', type=float, required=True)
    concept.add_argument('--ode-target', choices=['interval', 'instantaneous'], default='interval')
    concept.add_argument('--name', default='concept',
                         help='arm directory; a non-default name may join after the test')
    concept.add_argument('--allow-cpu', action='store_true')
    concept.set_defaults(func=concept_comparison)
    test = phases.add_parser('test', help='evaluate selected models on held-out test once')
    test.add_argument('--run', type=Path, required=True)
    test.add_argument('--test-data', type=Path, required=True)
    test.add_argument('--allow-cpu', action='store_true')
    test.set_defaults(func=test_comparison)
    extra = phases.add_parser('test-arm', help='test an extra named arm after the main test')
    extra.add_argument('--run', type=Path, required=True)
    extra.add_argument('--name', required=True)
    extra.add_argument('--test-data', type=Path, required=True)
    extra.add_argument('--allow-cpu', action='store_true')
    extra.set_defaults(func=test_arm)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
