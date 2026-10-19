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
        'interval': args.interval, 'train_steps': args.train_steps,
        'evaluation_steps': args.evaluation_steps, 'epochs': args.epochs,
        'train_stride': args.train_stride, 'validation_stride': args.validation_stride,
        'test_stride': args.test_stride, 'learning_rate': args.learning_rate,
        'seed': args.seed, 'scope': 'physics', 'forcing_mode': 'persistence',
        'tendency_weight': 0.0, 'acceleration_weight': 0.0, 'residual_parameter_weight': 1e-6,
        'selection': 'pooled multi-initialization validation normalized forecast MSE',
        'comparison_scope': 'transfer-learning experiment; not matched pretraining or parameter budgets',
    }
    save_json(output / 'protocol.json', protocol)
    for mode in modes:
        run_cli('init', '--mode', mode, '--checkpoint', args.checkpoint,
                '--data', args.train_data, '--seed', args.seed,
                '--output', output / 'checkpoints' / f'{mode}-init.pkl')
    # The original model is fine-tuned first, providing ONE train-derived scale
    # file that every architecture uses, including validation model selection.
    for mode in modes:
        label = 'baseline_finetuned' if mode == 'baseline' else mode
        command = [
            'train', '--checkpoint', output / 'checkpoints' / f'{mode}-init.pkl',
            '--train-data', args.train_data, '--validation-data', args.validation_data,
            '--steps', args.train_steps, '--interval', args.interval,
            '--validation-steps', args.evaluation_steps,
            '--validation-stride', args.validation_stride,
            '--stride', args.train_stride, '--epochs', args.epochs,
            '--learning-rate', args.learning_rate, '--seed', args.seed,
            '--scope', 'physics', '--forcing-mode', 'persistence',
            '--tendency-weight', 0, '--acceleration-weight', 0, '--residual-weight', 1e-6,
            '--output', output / label,
        ]
        if mode != 'baseline':
            command.extend(['--scales', output / 'baseline_finetuned' / 'scales.json'])
        run_cli(*command)
    protocol['phase'] = 'trained'
    save_json(output / 'protocol.json', protocol)
    print(f'Train and validation complete. Review {output}/protocol.json and each '
          'validation-selection.json before the separate test command.', flush=True)


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
    choices = {'baseline_pretrained': reference,
               'baseline_finetuned': output / 'baseline_finetuned' / 'best.pkl',
               'experts': output / 'experts' / 'best.pkl'}
    if 'memoryless' in protocol['modes']:
        choices['memoryless'] = output / 'memoryless' / 'best.pkl'
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
    for baseline in choices:
        if baseline == 'experts':
            continue
        subprocess.run([
            sys.executable, str(comparison), '--baseline', str(destination / baseline),
            '--candidate', str(destination / 'experts'),
            '--output', str(destination / f'experts_vs_{baseline}.csv'),
        ], check=True)
    protocol.update(phase='tested', test=info)
    save_json(output / 'protocol.json', protocol)
    print(f'Paired test results and improvement CSVs saved in {destination}', flush=True)


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
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--include-memoryless', action='store_true')
    train.add_argument('--allow-cpu', action='store_true')
    train.set_defaults(func=train_comparison)
    test = phases.add_parser('test', help='evaluate selected models on held-out test once')
    test.add_argument('--run', type=Path, required=True)
    test.add_argument('--test-data', type=Path, required=True)
    test.add_argument('--allow-cpu', action='store_true')
    test.set_defaults(func=test_comparison)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
