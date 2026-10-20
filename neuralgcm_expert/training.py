"""Differentiate the official coupled NeuralGCM rollout, with optional ODE losses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dinosaur import pytree_utils
from neuralgcm import PressureLevelModel

from neuralgcm_expert.losses import weighted_mse, floating_dynamic_fields


@dataclass(frozen=True)
class LossConfig:
    tendency_weight: float = 0.0
    acceleration_weight: float = 0.0
    residual_parameter_weight: float = 1e-6

    def __post_init__(self):
        for value in (self.tendency_weight, self.acceleration_weight,
                      self.residual_parameter_weight):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Loss weights must be finite and nonnegative")


def with_params(model, params):
    """Retain the official structure and public inference class."""
    result = PressureLevelModel(model._structure, params, model.gin_config)
    if hasattr(model, 'expert_metadata'):
        result.expert_metadata = model.expert_metadata
    if hasattr(model, '_checkpoint_aux_ds_dict'):
        result._checkpoint_aux_ds_dict = model._checkpoint_aux_ds_dict
    return result


def state_fields(state):
    """Floating prognostic fields; the clock is not a prediction target."""
    fields, _ = pytree_utils.as_dict(state)
    return {k: v for k, v in fields.items() if k != 'sim_time'}


def estimate_scales(dataset, variables):
    """Training-only normalization; callers must never pass validation/test data."""
    result = {}
    for name in variables:
        field = dataset[name]
        # Welford merge over bounded time chunks, never materialize multi-year ERA5.
        count, mean, m2 = 0, 0.0, 0.0
        slices = (field.isel(time=slice(i, i+4)) for i in range(0, field.sizes['time'], 4)) if 'time' in field.dims else (field,)
        for chunk in slices:
            values = np.asarray(chunk.values, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite training values in {name}")
            n, chunk_mean = values.size, float(values.mean())
            chunk_m2 = float(np.sum((values-chunk_mean)**2))
            delta = chunk_mean-mean
            combined = count+n
            m2 += chunk_m2 + delta*delta*count*n/combined
            mean += delta*n/combined
            count = combined
        scale = float(np.sqrt(m2/count))
        if scale <= 0:
            raise ValueError(f"Zero variance for {name}; supply a physical scale explicitly")
        result[name] = scale
    return result


def trainable_mask(params, scope='physics'):
    if scope not in {'physics', 'all'}:
        raise ValueError("scope must be physics or all")
    mask = {
        module: {name: scope == 'all' or 'div_curl_neural_parameterization' in module
                 for name in values}
        for module, values in params.items()
    }
    if not any(jax.tree_util.tree_leaves(mask)):
        raise ValueError("No learned-physics parameters matched the training scope")
    return mask


def residual_parameter_penalty(params):
    """Small parameter prior; this is explicitly NOT a physical noise estimate."""
    leaves = [value for module, values in params.items() if 'residual' in module
              for value in values.values()]
    if not leaves:
        return jnp.asarray(0.0)
    return sum(jnp.sum(x * x) for x in leaves) / sum(x.size for x in leaves)


def prepare_native_targets(model, dataset, output_interval, *, native_scales):
    """Encode a reference triplet once; targets are frozen during optimization.

    Finite differences describe the encoder's native coordinate representation,
    not observed process-specific tendencies. Units are nondimensional model time.
    """
    import pandas as pd
    if native_scales is None:
        raise ValueError("ODE losses require explicit native field scales")
    if dataset.sizes.get('time', 0) != 3:
        raise ValueError("Native derivative supervision requires exactly three times")
    times = dataset.time.values
    expected = np.timedelta64(pd.Timedelta(output_interval).value, 'ns')
    if not np.all(np.diff(times) == expected):
        raise ValueError("Native target triplet must have uniform output_interval spacing")
    states, forcings = [], []
    for index in range(3):
        sample = dataset.isel(time=index)
        inputs = model.inputs_from_xarray(sample)
        forcing = model.forcings_from_xarray(sample)
        states.append(model.encode(inputs, forcing))
        forcings.append(jax.tree_util.tree_map(jnp.asarray, forcing))
    dt = float(model.to_nondim_units(pd.Timedelta(output_interval).total_seconds(), 'second'))
    x0, x1, x2 = [s.state for s in states]
    first = jax.tree_util.tree_map(lambda a, b: (b-a)/(2*dt), x0, x2)
    second = jax.tree_util.tree_map(lambda a, b, c: (c-2*b+a)/(dt*dt), x0, x1, x2)
    required = set(floating_dynamic_fields(state_fields(first)))
    if set(native_scales) != required:
        raise ValueError(f"native_scales must specify every native field exactly; expected {sorted(required)}")
    if any(not np.isfinite(v) or v <= 0 for v in native_scales.values()):
        raise ValueError('native_scales must contain finite positive values')
    return {'state': states[1], 'forcings': forcings[1],
            'tendency': state_fields(first), 'acceleration': state_fields(second),
            'scales': native_scales}


def make_train_step(model, *, forecast_steps, output_interval, variable_scales,
                    spatial_weights=None, loss_config=LossConfig(),
                    learning_rate=1e-4, scope='physics', native_scales=None,
                    warmup_steps=0):
    """Return optimizer, state, and a jitted update using the real official solver.

    batch = {'inputs', 'initial_forcings', 'forcings', 'targets'} plus optional
    'native' from prepare_native_targets. No teacher forcing in the forecast.
    """
    if (not isinstance(forecast_steps, int) or isinstance(forecast_steps, bool)
            or forecast_steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("forecast_steps and learning_rate must be positive")
    use_native = bool(loss_config.tendency_weight or loss_config.acceleration_weight)
    if use_native and not native_scales:
        raise ValueError("Supply native_scales to enable native ODE losses")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    mask = trainable_mask(model.params, scope)
    labels = jax.tree_util.tree_map(lambda x: 'train' if x else 'freeze', mask)
    # Linear warmup to a constant peak, as in NeuralGCM's own optimizer schedule.
    schedule = (optax.linear_schedule(learning_rate / warmup_steps, learning_rate, warmup_steps)
                if warmup_steps else learning_rate)
    optimizer = optax.multi_transform({
        'train': optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule)),
        'freeze': optax.set_to_zero(),
    }, labels)

    def objective(params, batch):
        effective_params = jax.tree_util.tree_map(
            lambda value, enabled: value if enabled else jax.lax.stop_gradient(value),
            params, mask)
        current = with_params(model, effective_params)
        initial = current.encode(batch['inputs'], batch['initial_forcings'])
        _, prediction = current.unroll(initial, batch['forcings'],
                                       steps=forecast_steps, timedelta=output_interval,
                                       start_with_input=False)
        prediction = {k: prediction[k] for k in variable_scales}
        target = {k: batch['targets'][k] for k in variable_scales}
        forecast = weighted_mse(prediction, target, variable_scales=variable_scales,
                                spatial_weights=spatial_weights)
        regularization = residual_parameter_penalty(effective_params)
        tendency, acceleration = jnp.asarray(0.0), jnp.asarray(0.0)
        if use_native:
            from neuralgcm_expert.dynamics import native_rhs, native_acceleration
            native = batch['native']
            if loss_config.tendency_weight:
                tendency = weighted_mse(
                    state_fields(native_rhs(current, native['state'], native['forcings'])),
                    native['tendency'], variable_scales=native_scales)
            if loss_config.acceleration_weight:
                acceleration = weighted_mse(
                    state_fields(native_acceleration(current, native['state'], native['forcings'])),
                    native['acceleration'], variable_scales=native_scales)
        total = (forecast + loss_config.tendency_weight*tendency
                 + loss_config.acceleration_weight*acceleration
                 + loss_config.residual_parameter_weight*regularization)
        return total, {'total': total, 'forecast': forecast, 'tendency': tendency,
                       'acceleration': acceleration, 'residual_parameter': regularization}

    @jax.jit
    def update(params, opt_state, batch):
        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params, batch)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        finite = jnp.isfinite(metrics['total']) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(
                (gradients, next_params, next_opt_state))]))
        # Rejected updates preserve both model and optimizer state.
        next_params, next_opt_state = jax.lax.cond(
            finite, lambda _: (next_params, next_opt_state),
            lambda _: (params, opt_state), operand=None)
        metrics = {**metrics, 'accepted': finite}
        return next_params, next_opt_state, metrics

    return optimizer, optimizer.init(model.params), update


def window_batch(window):
    return jax.tree_util.tree_map(jnp.asarray, {
        'inputs': window.inputs, 'initial_forcings': window.initial_forcings,
        'forcings': window.forcings, 'targets': window.targets})
