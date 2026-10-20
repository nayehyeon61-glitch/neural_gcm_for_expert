"""Continuous coupled tendencies of the *actual* NeuralGCM dynamical core.

The forecast is still advanced by NeuralGCM's original IMEX integrator. These
functions expose its instantaneous semidiscrete field for derivative losses:
``F(x,t) = explicit_dycore(x) + implicit_dycore(x) + learned_physics(x,t)``.
The original parameterization (including its tendency transforms/filter) is
called with the original Haiku parameter paths. Numerical *step* filters and
the finite-step IMEX map are not continuous tendencies and are excluded.

State, tendencies, and time all use the checkpoint's native nondimensional
units. Memory models: the physics network sees ``encoded_state.memory`` (the previous
state used by memory features), held fixed; F is the tendency at that memory.
The return value is the prognostic state tree, not a ModelState carrying
diagnostics or random-generator bookkeeping. In particular, F.sim_time = 1.
"""

from __future__ import annotations

from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
from dinosaur import typing as dinosaur_typing
from neuralgcm.legacy import api, correctors, diagnostics, gin_utils
from neuralgcm.legacy import perturbations, steps, stochastic


class UnsupportedNativeDynamicsError(ValueError):
    """The checkpoint cannot be represented by the supported memoryless RHS."""


def _none_is_leaf(value: Any) -> bool:
    return value is None


def _complete_tendency(reference: Any, tendency: Any) -> Any:
    """Replace masked/None derivatives by zeros using the physical state tree."""
    def complete(x, dx):
        if x is None:
            if dx is not None:
                raise ValueError("A tendency exists for an absent state leaf.")
            return None
        return jnp.zeros_like(x) if dx is None else jnp.asarray(dx, dtype=jnp.asarray(x).dtype)

    return jax.tree_util.tree_map(
        complete, reference, tendency, is_leaf=_none_is_leaf
    )


def _sum_tendencies(*trees: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda *xs: None if xs[0] is None else sum(xs),
        *trees,
        is_leaf=_none_is_leaf,
    )


def _corrector_components(corrector: Any, state: Any, physics: Any) -> dict[str, Any]:
    """Pull the native dycore field back through linear spectral interpolation."""
    if isinstance(corrector, correctors.CustomCoordsCorrector):
        inner = _corrector_components(
            corrector.corrector_fn,
            corrector.to_custom_coords_fn(state),
            corrector.to_custom_coords_fn(physics),
        )
        return {
            name: corrector.from_custom_coords_fn(value)
            for name, value in inner.items()
        }
    if not isinstance(corrector, correctors.DycoreWithPhysicsCorrector):
        raise UnsupportedNativeDynamicsError(
            "Continuous RHS supports DycoreWithPhysicsCorrector, optionally "
            "wrapped in CustomCoordsCorrector; got " + type(corrector).__name__
        )
    state, physics = corrector.coords.with_dycore_sharding((state, physics))
    equation = corrector.dycore_equation
    result = {
        "dycore_explicit": _complete_tendency(state, equation.explicit_terms(state)),
        "dycore_implicit": _complete_tendency(state, equation.implicit_terms(state)),
        "learned_physics": _complete_tendency(state, physics),
    }
    result["total"] = _sum_tendencies(*result.values())
    return result


def _parameterization_step(model_module: Any) -> Any:
    step = model_module.advance_fn
    # RepeatedStep changes the integration cadence, not the instantaneous field.
    while isinstance(step, steps.RepeatedStep):
        step = step.step_fn
    if not isinstance(step, steps.StochasticPhysicsParameterizationStep):
        raise UnsupportedNativeDynamicsError(
            "Continuous RHS requires StochasticPhysicsParameterizationStep "
            "(with randomness disabled); got " + type(step).__name__
        )
    if not isinstance(step.randomness_fn, (stochastic.NoRandomField, stochastic.ZerosRandomField)):
        raise UnsupportedNativeDynamicsError("Disable stochastic forcing before evaluating the ODE RHS.")
    if not isinstance(step.perturbation_fn, perturbations.NoPerturbation):
        raise UnsupportedNativeDynamicsError("Continuous RHS requires NoPerturbation.")
    if not isinstance(step.diagnostics_fn, diagnostics.NoDiagnostics):
        raise UnsupportedNativeDynamicsError(
            "Stateful diagnostics are unsupported; use NoDiagnostics for a closed memoryless ODE."
        )
    return step


def _prognostic_state(encoded_state: Any) -> Any:
    if not isinstance(encoded_state, dinosaur_typing.ModelState):
        raise TypeError("encoded_state must be the ModelState returned by model.encode().")
    if getattr(encoded_state.state, "sim_time", None) is None:
        raise ValueError("The encoded physical state must include nondimensional sim_time.")
    return encoded_state.state


def _processed_forcings(model: Any, state: Any, forcings: Any, params: Any) -> Any:
    # Exactly the public PressureLevelModel.advance preprocessing; all bindings
    # are evaluated under the caller's specific_config context.
    forcings = model._squeeze_level_from_forcings(forcings)
    forcings = api._prepend_dummy_time_axis(forcings)
    return model._structure.forcing_fn(params, None, forcings, state.sim_time)


def _transformed_components(model: Any) -> hk.Transformed:
    def forward(state, memory, forcing):
        # Construct the full original model: directly constructing a standalone
        # parameterization would change Haiku names and miss checkpoint weights.
        module = model._structure.model_cls()
        step = _parameterization_step(module)
        randomness = step.randomness_fn.unconditional_sample(None).nodal_value
        physics = step.physics_parameterization_fn(
            state, memory, {}, randomness, forcing
        )
        return _corrector_components(step.corrector_fn, state, physics)

    return hk.without_apply_rng(hk.transform(forward))


def native_rhs_components(
    model: Any, encoded_state: Any, forcings: Any, *, params: Any = None
) -> dict[str, Any]:
    """Return native explicit/implicit/learned/total coupled tendencies.

    ``forcings`` has the same single-time raw pressure-level format accepted by
    ``model.advance``. ``params`` can override the model parameters for training.
    CustomCoordsCorrector contributes ``P_back F_inner(P_forward x)``. This
    projected field is not a claim that a finite filtered IMEX step equals its
    exact continuous flow.
    """
    state = _prognostic_state(encoded_state)
    params = model.params if params is None else params
    with gin_utils.specific_config(model.gin_config):
        forcing = _processed_forcings(model, state, forcings, params)
        return _transformed_components(model).apply(
            params, state, encoded_state.memory, forcing)


def native_rhs(model: Any, encoded_state: Any, forcings: Any, *, params: Any = None) -> Any:
    """Return F on native sigma/modal coordinates, in nondimensional units."""
    return native_rhs_components(model, encoded_state, forcings, params=params)["total"]


def material_derivative(field: Callable[[Any], Any], state: Any) -> Any:
    """Return ``D field(state)[field(state)]`` without constructing a Jacobian.

    For a nonautonomous ODE, include time in the state and return dt/dt=1 in
    ``field``. This includes explicit time dependence as well as all variable
    cross derivatives. None leaves are supported by JAX pytree semantics.
    """
    velocity = field(state)
    return jax.jvp(field, (state,), (velocity,))[1]


def native_acceleration(
    model: Any,
    encoded_state: Any,
    forcings: Any,
    *,
    params: Any = None,
    processed_forcing_tangent: Any = None,
) -> Any:
    """Differentiate the coupled native RHS along its own trajectory.

    Computes ``D_x F[F] + partial_t F`` using an augmented state with
    ``sim_time'=1``. External forcing is selected once at the reference time
    and held fixed, matching NeuralGCM's step convention. Time-dependent solar
    and orbital features remain differentiated through the state sim_time.

    Optionally pass ``processed_forcing_tangent = d f / d t`` in the original
    model's *processed* forcing tree and native time units to add ``D_f F[f']``.
    This is deliberately explicit: nearest-time forcing selection is not a
    differentiable continuous forcing model. ``params`` remains differentiable
    through the JVP, so this function can be used in a training loss.
    """
    state = _prognostic_state(encoded_state)
    params = model.params if params is None else params
    with gin_utils.specific_config(model.gin_config):
        forcing = _processed_forcings(model, state, forcings, params)
        apply = _transformed_components(model).apply

        def field(x, f):
            return apply(params, x, encoded_state.memory, f)["total"]

        velocity = field(state, forcing)
        if processed_forcing_tangent is None:
            return jax.jvp(lambda x: field(x, forcing), (state,), (velocity,))[1]
        return jax.jvp(
            field,
            (state, forcing),
            (velocity, processed_forcing_tangent),
        )[1]


# Short aliases are useful in experiment code without changing native semantics.
rhs = native_rhs
acceleration = native_acceleration
