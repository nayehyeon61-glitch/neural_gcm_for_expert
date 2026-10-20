"""Differentiate the official coupled NeuralGCM rollout, with optional ODE losses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dinosaur import pytree_utils
from neuralgcm import PressureLevelModel

from neuralgcm_expert.losses import weighted_mse, floating_dynamic_fields


ODE_GROUPS = ('temperature', 'momentum', 'moisture')


def ode_group(name):
    """Physical concept of a native state field; None for fields physics does not set."""
    parts = name.split('/')
    if 'tracers' in parts:
        return 'moisture'
    if parts[-1] == 'temperature_variation':
        return 'temperature'
    if parts[-1] in ('vorticity', 'divergence'):
        return 'momentum'
    return None  # log_surface_pressure: masked in learned physics, dycore only


@dataclass(frozen=True)
class LossConfig:
    tendency_weight: float = 0.0
    acceleration_weight: float = 0.0
    residual_parameter_weight: float = 1e-6
    # Per-concept intermediate ODE losses: lambda_k for temperature/momentum/moisture.
    ode_weights: Mapping[str, float] = field(default_factory=dict)
    # 'interval': solver-integrated (core + learned) concept tendency over each
    # output interval vs the encoded data; 'instantaneous': native RHS vs the
    # central difference (only meaningful for short intervals, e.g. hourly data).
    ode_target: str = 'interval'

    def __post_init__(self):
        for value in (self.tendency_weight, self.acceleration_weight,
                      self.residual_parameter_weight, *self.ode_weights.values()):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Loss weights must be finite and nonnegative")
        if set(self.ode_weights) - set(ODE_GROUPS):
            raise ValueError(f"ode_weights keys must be among {ODE_GROUPS}")
        if self.ode_target not in ('interval', 'instantaneous'):
            raise ValueError("ode_target must be 'interval' or 'instantaneous'")

    @property
    def uses_native(self):
        return bool(self.tendency_weight or self.acceleration_weight
                    or any(self.ode_weights.values()))


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


def prepare_native_targets(model, dataset, output_interval, *, native_scales,
                           validate_scales=True):
    """Encode a reference triplet once; targets are frozen during optimization.

    Finite differences describe the encoder's native coordinate representation,
    not observed process-specific tendencies. Units are nondimensional model time.
    """
    import pandas as pd
    if validate_scales and native_scales is None:
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
    if validate_scales:
        if set(native_scales) != required:
            raise ValueError(f"native_scales must specify every native field exactly; expected {sorted(required)}")
        if any(not np.isfinite(v) or v <= 0 for v in native_scales.values()):
            raise ValueError('native_scales must contain finite positive values')
    return {'state': states[1], 'forcings': forcings[1],
            'tendency': state_fields(first), 'acceleration': state_fields(second),
            'encoded': [state_fields(x) for x in (x0, x1, x2)],
            'scales': native_scales}


def checkpointed_unroll(model, state, forcings, steps, output_interval, return_states=False):
    """Same values as model.unroll, with each output interval rematerialized.

    The official unroll keeps every inner model step of the whole rollout for
    the backward pass, so memory grows with the forecast length (one 24h step
    on L40S fits; two do not). Saving only one state per output interval keeps
    multi-step training at roughly the single-step footprint.
    """
    if steps == 1:
        state, outputs = model.unroll(state, forcings, steps=1, timedelta=output_interval,
                                      start_with_input=False)
        return (outputs, [state.state]) if return_states else outputs
    advance = jax.checkpoint(lambda s: model.unroll(
        s, forcings, steps=1, timedelta=output_interval, start_with_input=False))
    outputs, states = [], []
    for _ in range(steps):
        state, output = advance(state)
        outputs.append(output)
        states.append(state.state)
    outputs = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outputs)
    return (outputs, states) if return_states else outputs


def estimate_native_scales(model, dataset, output_interval, start_indices, target='interval'):
    """RMS of the data-derived native tendency per field, over training triplets only.

    ``target`` matches the concept loss: one-interval difference or central difference.

    Native fields are modal coefficients in nondimensional units, so data-space
    scales do not apply; these fixed scales normalize each concept ODE loss.
    """
    import pandas as pd
    interval = np.timedelta64(pd.Timedelta(output_interval).value, 'ns')
    times = dataset.time.values
    sums, count = {}, 0
    for index in start_indices:
        triplet = dataset.sel(time=times[index] + np.arange(3)*interval)
        native = prepare_native_targets(model, triplet, output_interval, native_scales=None,
                                        validate_scales=False)
        tendency = native['tendency']
        if target == 'interval':
            span = float(model.to_nondim_units(pd.Timedelta(output_interval).total_seconds(), 'second'))
            tendency = jax.tree_util.tree_map(lambda a, b: (b-a)/span, *native['encoded'][:2])
        for name, value in floating_dynamic_fields(tendency).items():
            sums[name] = sums.get(name, 0.0) + float(jnp.mean(jnp.square(value)))
        count += 1
    if not count:
        raise ValueError('No training triplets for native scale estimation')
    scales = {name: float(np.sqrt(total/count)) for name, total in sums.items()}
    if any(not np.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError(f'Degenerate native tendency scales: {scales}')
    return scales


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
    use_native = loss_config.uses_native
    concept = any(loss_config.ode_weights.values())
    if concept and loss_config.ode_target == 'interval' and forecast_steps > 2:
        raise ValueError('interval concept loss uses the encoded triplet: forecast_steps <= 2')
    import pandas as pd
    interval_nondim = float(model.to_nondim_units(
        pd.Timedelta(output_interval).total_seconds(), 'second'))
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
        prediction, rolled = checkpointed_unroll(current, initial, batch['forcings'],
                                                 forecast_steps, output_interval,
                                                 return_states=True)
        prediction = {k: prediction[k] for k in variable_scales}
        target = {k: batch['targets'][k] for k in variable_scales}
        forecast = weighted_mse(prediction, target, variable_scales=variable_scales,
                                spatial_weights=spatial_weights)
        regularization = residual_parameter_penalty(effective_params)
        tendency, acceleration = jnp.asarray(0.0), jnp.asarray(0.0)
        concept_losses = {}
        if use_native:
            from neuralgcm_expert.dynamics import native_rhs, native_acceleration
            native = batch['native']
            instantaneous = loss_config.ode_target == 'instantaneous'
            if loss_config.tendency_weight or (instantaneous and concept):
                rhs = state_fields(native_rhs(current, native['state'], native['forcings']))
            if loss_config.tendency_weight:
                tendency = weighted_mse(rhs, native['tendency'], variable_scales=native_scales)
            if concept:
                # Core + learned tendency of each concept vs the data-derived total
                # tendency of the same fields; the supervised values enter the solver.
                if instantaneous:
                    pairs = [(floating_dynamic_fields(rhs),
                              floating_dynamic_fields(native['tendency']))]
                else:
                    origin = floating_dynamic_fields(native['encoded'][0])
                    pairs = []
                    for j, x_hat in enumerate(rolled, start=1):
                        span = j*interval_nondim
                        x_hat = floating_dynamic_fields(state_fields(x_hat))
                        x_ref = floating_dynamic_fields(native['encoded'][j])
                        pairs.append(({n: (x_hat[n]-origin[n])/span for n in origin},
                                      {n: (x_ref[n]-origin[n])/span for n in origin}))
                for group, weight in loss_config.ode_weights.items():
                    names = [n for n in pairs[0][0] if ode_group(n) == group]
                    if weight and names:
                        concept_losses[group] = sum(weighted_mse(
                            {n: predicted[n] for n in names}, {n: expected[n] for n in names},
                            variable_scales={n: native_scales[n] for n in names})
                            for predicted, expected in pairs) / len(pairs)
            if loss_config.acceleration_weight:
                acceleration = weighted_mse(
                    state_fields(native_acceleration(current, native['state'], native['forcings'])),
                    native['acceleration'], variable_scales=native_scales)
        total = (forecast + loss_config.tendency_weight*tendency
                 + loss_config.acceleration_weight*acceleration
                 + loss_config.residual_parameter_weight*regularization
                 + sum(loss_config.ode_weights[g]*value for g, value in concept_losses.items()))
        return total, {'total': total, 'forecast': forecast, 'tendency': tendency,
                       'acceleration': acceleration, 'residual_parameter': regularization,
                       **{f'ode_{g}': value for g, value in concept_losses.items()}}

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
