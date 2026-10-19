"""Differentiable trajectory and coupled-vector-field losses.

All derivatives are in the coordinates and time units supplied by the caller.
In particular, use nondimensional time for NeuralGCM's encoded modal state.
These helpers do not replace NeuralGCM's integrator or evaluation protocol.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


def _named_tree(tree: PyTree) -> PyTree:
    # tree_math/Dinosaur registers dataclasses with positional pytree keys.
    # Recover their real field names before constructing loss paths, otherwise
    # sim_time would appear as "5" and silently enter loss/scale calculations.
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        return {field.name: _named_tree(getattr(tree, field.name)) for field in dataclasses.fields(tree)}
    if isinstance(tree, Mapping):
        return {key: _named_tree(value) for key, value in tree.items()}
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return {name: _named_tree(getattr(tree, name)) for name in tree._fields}
    if isinstance(tree, (tuple, list)):
        return type(tree)(_named_tree(value) for value in tree)
    return tree


def _path_parts(path: tuple[Any, ...]) -> tuple[str, ...]:
    parts = []
    for key in path:
        for attribute in ("key", "name", "idx"):
            if hasattr(key, attribute):
                parts.append(str(getattr(key, attribute)))
                break
        else:
            parts.append(str(key))
    return tuple(parts)


def floating_dynamic_fields(
    tree: PyTree, *, exclude_fields: Sequence[str] = ("sim_time",)
) -> dict[str, jax.Array]:
    """Flatten floating leaves by path; exclude clocks/counters from losses.

    Works with nested dictionaries and registered JAX pytrees, including
    Dinosaur state objects. Keys use slash-separated paths, e.g.
    ``tracers/specific_humidity``. Integer, boolean, None and float0 leaves are
    omitted. Floating ``sim_time`` is excluded from loss, but is NOT removed
    from :func:`total_derivative`, where its tangent may be essential.
    """
    result = {}
    for path, value in jax.tree_util.tree_flatten_with_path(_named_tree(tree))[0]:
        parts = _path_parts(path)
        name = "/".join(parts) or "value"
        if any(part in exclude_fields for part in parts):
            continue
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            continue
        if name in result:
            raise ValueError(f"Ambiguous flattened field path: {name}")
        result[name] = array
    return result


def _lookup(config: Mapping | None, name: str, default: Any) -> Any:
    if config is None:
        return default
    for key in (name, tuple(name.split("/")), name.split("/")[-1]):
        if key in config:
            return config[key]
    return default


def weighted_mse(
    prediction: PyTree,
    target: PyTree,
    *,
    variable_scales: Mapping | None = None,
    variable_weights: Mapping | None = None,
    spatial_weights: Any = None,
    mask: Any = None,
    exclude_fields: Sequence[str] = ("sim_time",),
) -> jax.Array:
    """Average scaled per-variable MSE, with optional quadrature and masks.

    ``variable_scales`` are positive fixed training-data scales, not variances.
    ``variable_weights`` are fixed nonnegative scalar weights. Both accept
    slash-separated paths, path tuples, or leaf names. ``spatial_weights`` and
    ``mask`` can be shared arrays or dictionaries keyed in the same way. Arrays
    must broadcast to the relevant field; for official decoded ``(..., lon,
    lat)`` arrays, a latitude weight vector broadcasts directly. Grid quadrature
    weights do not apply to modal coefficients: transform to nodal space first.

    Each field is normalized by its own total valid point weight; then active
    field losses are averaged using variable weights. Fully masked fields do
    not affect the denominator. Masked nonfinite values are ignored; nonfinite
    unmasked values propagate, so failures are not silently hidden. The function
    is JIT/gradient compatible; scales and variable weights are static settings.
    """
    predicted = floating_dynamic_fields(prediction, exclude_fields=exclude_fields)
    expected = floating_dynamic_fields(target, exclude_fields=exclude_fields)
    if predicted.keys() != expected.keys():
        raise ValueError(
            "Prediction and target floating field paths must match: "
            f"{sorted(predicted)} != {sorted(expected)}"
        )
    if not predicted:
        raise ValueError("No floating dynamic fields were selected for the loss")
    numerator = jnp.asarray(0.0)
    denominator = jnp.asarray(0.0)
    for name, prediction_value in predicted.items():
        target_value = expected[name]
        if prediction_value.shape != target_value.shape:
            raise ValueError(f"Shape mismatch for {name}: {prediction_value.shape} != {target_value.shape}")
        scale = float(_lookup(variable_scales, name, 1.0))
        field_weight = float(_lookup(variable_weights, name, 1.0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Scale for {name} must be finite and positive")
        if not np.isfinite(field_weight) or field_weight < 0:
            raise ValueError(f"Weight for {name} must be finite and nonnegative")
        if field_weight == 0:
            continue
        quadrature = _lookup(spatial_weights, name, 1.0) if isinstance(spatial_weights, Mapping) else spatial_weights
        valid_mask = _lookup(mask, name, 1.0) if isinstance(mask, Mapping) else mask
        quadrature = 1.0 if quadrature is None else quadrature
        valid_mask = 1.0 if valid_mask is None else valid_mask
        point_weight = jnp.broadcast_to(jnp.asarray(quadrature) * jnp.asarray(valid_mask), prediction_value.shape)
        positive = point_weight > 0
        # Mask before subtraction/squaring to avoid NaN gradients at missing data.
        safe_prediction = jnp.where(positive, prediction_value, 0)
        safe_target = jnp.where(positive, target_value, 0)
        squared_error = jnp.square((safe_prediction - safe_target) / scale)
        valid_weight = jnp.where(positive, point_weight, 0)
        point_total = jnp.sum(valid_weight)
        field_loss = jnp.sum(valid_weight * squared_error) / jnp.maximum(point_total, 1e-30)
        numerator = numerator + field_weight * field_loss
        denominator = denominator + field_weight * (point_total > 0)
    return numerator / jnp.maximum(denominator, 1e-30)


rollout_mse = weighted_mse


def _floating_jvp(function: Callable, primal: PyTree, tangent: PyTree):
    """JVP through floating primal leaves; keep integer metadata fixed."""
    primal_leaves, structure = jax.tree_util.tree_flatten(primal)
    tangent_leaves, tangent_structure = jax.tree_util.tree_flatten(tangent)
    if tangent_structure != structure:
        raise ValueError("A vector-field tangent must have the same pytree structure as its state")
    indices = [i for i, leaf in enumerate(primal_leaves) if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)]
    if not indices:
        raise ValueError("JVP requires at least one floating state leaf")

    def selected_function(selected):
        leaves = list(primal_leaves)
        for index, value in zip(indices, selected):
            leaves[index] = value
        return function(jax.tree_util.tree_unflatten(structure, leaves))

    selected_primals = tuple(jnp.asarray(primal_leaves[i]) for i in indices)
    selected_tangents = tuple(jnp.asarray(tangent_leaves[i], dtype=selected_primals[k].dtype) for k, i in enumerate(indices))
    return jax.jvp(selected_function, (selected_primals,), (selected_tangents,))


def total_derivative(
    vector_field: Callable[[PyTree, PyTree], PyTree],
    state: PyTree,
    forcing: PyTree = None,
    forcing_rate: PyTree = None,
) -> PyTree:
    r"""Return ``D_x F(x,c)[F(x,c)] + D_c F(x,c)[c_dot]``.

    Differentiate the assembled vector field, preserving all cross-variable
    terms. This is an auxiliary derivative of the continuous RHS; it is not a
    replacement for, or the exact acceleration of, a discrete IMEX trajectory.
    If ``forcing_rate`` is omitted, forcing is held fixed. Explicit time may be
    included as a floating ``sim_time`` state with RHS equal to one; it must NOT
    be masked out of this JVP. Nonfloating metadata is held constant.
    """
    velocity = vector_field(state, forcing)
    _, derivative = _floating_jvp(lambda x: vector_field(x, forcing), state, velocity)
    if forcing_rate is not None:
        if forcing is None:
            raise ValueError("forcing_rate requires a forcing pytree")
        _, forcing_derivative = _floating_jvp(lambda c: vector_field(state, c), forcing, forcing_rate)

        def add(a, b):
            return a + b if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a

        derivative = jax.tree_util.tree_map(add, derivative, forcing_derivative)
    return derivative


def tendency_matching_loss(vector_field, state, target_tendency, *, forcing=None, **mse_kwargs):
    """Match the full assembled RHS to a tendency target in the same units."""
    return weighted_mse(vector_field(state, forcing), target_tendency, **mse_kwargs)


def coupled_derivative_loss(
    vector_field, state, target_acceleration, *, forcing=None, forcing_rate=None, **mse_kwargs
):
    """Match full coupled RHS derivative to observed/teacher acceleration."""
    acceleration = total_derivative(vector_field, state, forcing, forcing_rate)
    return weighted_mse(acceleration, target_acceleration, **mse_kwargs)


acceleration_matching_loss = coupled_derivative_loss


class FiniteDifferenceTargets(NamedTuple):
    tendency: dict[str, jax.Array]
    acceleration: dict[str, jax.Array]


def validate_uniform_timestep(
    dt: float | None = None,
    times: Any = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> float:
    """Validate static numeric times and return their positive uniform spacing.

    Call during data preparation, outside ``jax.jit``. Times and ``dt`` must
    use the vector field's time units. Datetimes must be converted explicitly.
    """
    if times is not None:
        values = np.asarray(times)
        if values.ndim != 1 or values.size < 3 or not np.issubdtype(values.dtype, np.number):
            raise ValueError("times must be a numeric 1D array with at least three entries")
        differences = np.diff(values.astype(np.float64))
        observed = float(differences[0])
        if not np.all(np.isfinite(values)) or observed <= 0 or not np.allclose(differences, observed, rtol=rtol, atol=atol):
            raise ValueError("Finite differences require finite, increasing, uniformly spaced times")
        if dt is not None and not np.isclose(float(dt), observed, rtol=rtol, atol=atol):
            raise ValueError("dt disagrees with the supplied times")
        dt = observed
    if dt is None or not np.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("dt must be finite and positive, or provide uniform times")
    return float(dt)


def finite_difference_targets(
    trajectory: PyTree,
    dt: float | None = None,
    *,
    times: Any = None,
    time_axis: int = 0,
    exclude_fields: Sequence[str] = ("sim_time",),
) -> FiniteDifferenceTargets:
    """Centered tendency/acceleration targets at interior samples ``1:-1``.

    Returns flattened floating field dictionaries, preserving the requested
    time-axis position and excluding ``sim_time``. All fields must share their
    sample count. For tendency use (x[t+1]-x[t-1])/(2 dt), for acceleration use
    (x[t+1]-2 x[t]+x[t-1])/dt**2. Finite sampling and analysis increments make
    these approximate targets; they are not physical process labels.
    """
    step = validate_uniform_timestep(dt, times)
    fields = floating_dynamic_fields(trajectory, exclude_fields=exclude_fields)
    if not fields:
        raise ValueError("No floating dynamic trajectory fields were selected")
    tendency, acceleration = {}, {}
    sample_count = None
    for name, value in fields.items():
        if value.ndim == 0 or not -value.ndim <= time_axis < value.ndim:
            raise ValueError(f"Invalid time axis for {name}")
        axis = time_axis % value.ndim
        count = value.shape[axis]
        if count < 3:
            raise ValueError("At least three trajectory samples are required")
        if sample_count is not None and count != sample_count:
            raise ValueError("All trajectory fields must share the time sample count")
        if times is not None and count != len(times):
            raise ValueError("times length does not match trajectory sample count")
        sample_count = count
        moved = jnp.moveaxis(value, axis, 0)
        first = (moved[2:] - moved[:-2]) / (2 * step)
        second = (moved[2:] - 2 * moved[1:-1] + moved[:-2]) / step**2
        tendency[name] = jnp.moveaxis(first, 0, axis)
        acceleration[name] = jnp.moveaxis(second, 0, axis)
    return FiniteDifferenceTargets(tendency, acceleration)


def residual_size_penalty(residual: PyTree, **mse_kwargs) -> jax.Array:
    """Optional scaled residual energy; does not identify stochastic noise."""
    residual = floating_dynamic_fields(residual)
    return weighted_mse(residual, jax.tree_util.tree_map(jnp.zeros_like, residual), **mse_kwargs)
