"""Native RHS tests, including a tiny real Dinosaur primitive-equation core."""
from __future__ import annotations

import dataclasses
import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dinosaur import coordinate_systems, primitive_equations, sigma_coordinates
from dinosaur import spherical_harmonic, typing
from neuralgcm.legacy import api, correctors, equations, model_builder, steps, stochastic

from neuralgcm_expert.dynamics import (
    UnsupportedNativeDynamicsError,
    material_derivative,
    native_acceleration,
    native_rhs,
    native_rhs_components,
)


def test_material_derivative_includes_cross_coupling_and_explicit_time():
    # T' = w + t, w' = -2T. Thus T'' = -2T + 1 and w'' = -2(w+t).
    state = {"T": jnp.array(3.0), "wind": jnp.array(4.0), "sim_time": jnp.array(2.0), "none": None}
    def field(x):
        return {"T": x["wind"] + x["sim_time"], "wind": -2*x["T"], "sim_time": jnp.ones_like(x["sim_time"]), "none": None}
    result = material_derivative(field, state)
    np.testing.assert_allclose(result["T"], -5)
    np.testing.assert_allclose(result["wind"], -12)
    np.testing.assert_allclose(result["sim_time"], 0)
    assert result["none"] is None


class _TinyPhysics(hk.Module):
    """Small learned test tendency; the dynamical equations remain upstream."""
    def __init__(self, coords, dt, physics_specs, aux_features):
        super().__init__()

    def __call__(self, state, memory, diagnostics, randomness, forcing):
        gain = hk.get_parameter("gain", (), init=hk.initializers.Constant(0.25))
        temperature = gain * state.temperature_variation
        temperature = temperature.at[:, 0, 0].add(0.001 * state.sim_time)
        return dataclasses.replace(
            state,
            vorticity=jnp.zeros_like(state.vorticity),
            divergence=jnp.zeros_like(state.divergence),
            temperature_variation=temperature,
            log_surface_pressure=None,
            tracers=jax.tree_util.tree_map(jnp.zeros_like, state.tracers),
            sim_time=None,
        )


@pytest.fixture(scope="module", params=[False, True], ids=["native", "custom-dycore-coords"])
def native_fixture(request):
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid(
            longitude_wavenumbers=3, total_wavenumbers=4,
            longitude_nodes=8, latitude_nodes=4, radius=physics_specs.radius,
        ),
        sigma_coordinates.SigmaCoordinates.equidistant(2),
    )
    # Genuine NeuralGCM corrector + genuine primitive equations. Initialize by
    # invoking the original model.advance path, then reuse those trained names.
    dycore = functools.partial(
        equations.PrimitiveEquationsWithTime,
        orography_module=lambda coords, *args: lambda: jnp.zeros(coords.horizontal.modal_shape),
    )
    corrector_module = functools.partial(
        correctors.DycoreWithPhysicsCorrector,
        dycore_equation_module=dycore,
        dycore_substeps=1,
        checkpoint_explicit_terms=False,
    )
    if request.param:
        custom_coords = dataclasses.replace(
            coords,
            horizontal=spherical_harmonic.Grid(
                longitude_wavenumbers=4, total_wavenumbers=5,
                longitude_nodes=12, latitude_nodes=6, radius=physics_specs.radius,
            ),
        )
        corrector_module = functools.partial(
            correctors.CustomCoordsCorrector,
            corrector_module=corrector_module,
            custom_coords=custom_coords,
        )
    advance_module = functools.partial(
        steps.StochasticPhysicsParameterizationStep,
        corrector_module=corrector_module,
        physics_parameterization_module=_TinyPhysics,
        randomness_module=stochastic.NoRandomField,
    )
    model_cls = functools.partial(
        model_builder.StochasticModularStepModel,
        advance_module=advance_module,
        encoder_module=lambda *args: None,
        decoder_module=lambda *args: None,
    )
    structure = model_builder.WhirlModel(
        coords, 0.01, physics_specs,
        {"ref_temperatures": np.array([288.0, 288.0])},
        model_cls=model_cls,
    )
    zeros = jnp.zeros(coords.modal_shape)
    state = primitive_equations.StateWithTime(
        vorticity=zeros,
        divergence=zeros,
        temperature_variation=zeros.at[:, 0, 0].set(0.2),
        log_surface_pressure=jnp.zeros((1,) + coords.horizontal.modal_shape),
        tracers={"specific_humidity": zeros},
        sim_time=jnp.array(2.0),
    )
    random_state = stochastic.NoRandomField(coords, 0.01, physics_specs, {}).unconditional_sample(None)
    encoded = typing.ModelState(state=state, randomness=random_state)
    transformed = hk.transform(lambda x, f: structure.model_cls().advance(x, f))
    params = transformed.init(jax.random.key(0), encoded, {})
    model = api.PressureLevelModel(structure, params, "")
    return model, encoded


def test_native_rhs_uses_original_advance_parameter_paths_and_all_terms(native_fixture):
    model, encoded = native_fixture
    assert any("stochastic_modular_step_model" in name for name in model.params)
    result = native_rhs_components(model, encoded, {})
    assert set(result) == {"dycore_explicit", "dycore_implicit", "learned_physics", "total"}
    np.testing.assert_allclose(result["total"].sim_time, 1)
    np.testing.assert_allclose(result["learned_physics"].sim_time, 0)
    # Native masked pressure tendency becomes an actual zero tangent.
    np.testing.assert_allclose(result["learned_physics"].log_surface_pressure, 0)
    np.testing.assert_allclose(result["learned_physics"].temperature_variation[:, 0, 0], 0.052)
    for total, explicit, implicit, physics in zip(*(
        jax.tree_util.tree_leaves(result[name])
        for name in ("total", "dycore_explicit", "dycore_implicit", "learned_physics")
    )):
        np.testing.assert_allclose(total, explicit + implicit + physics, rtol=1e-6, atol=1e-7)


def test_native_acceleration_time_and_parameter_gradients(native_fixture):
    model, encoded = native_fixture
    acceleration = native_acceleration(model, encoded, {})
    np.testing.assert_allclose(acceleration.sim_time, 0)
    # Uniform atmosphere has zero dry-core motion: 0.25*0.052 + 0.001.
    np.testing.assert_allclose(acceleration.temperature_variation[:, 0, 0], 0.014, rtol=1e-4, atol=1e-6)
    derivative = jax.grad(lambda params: jnp.sum(
        native_acceleration(model, encoded, {}, params=params).temperature_variation[:, 0, 0]
    ))(model.params)
    gain_derivatives = [values["gain"] for values in derivative.values() if "gain" in values]
    # Two layers: d/d gain [gain*(gain*0.2+0.002)+0.001] = 0.204.
    np.testing.assert_allclose(gain_derivatives, [0.204], rtol=1e-4, atol=1e-6)


def test_memory_cannot_silently_enter_native_rhs(native_fixture):
    model, encoded = native_fixture
    with pytest.raises(UnsupportedNativeDynamicsError, match="memory=None"):
        native_rhs(model, dataclasses.replace(encoded, memory=encoded.state), {})
