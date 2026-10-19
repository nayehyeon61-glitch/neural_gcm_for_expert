import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralgcm_expert.losses import (
    coupled_derivative_loss,
    finite_difference_targets,
    floating_dynamic_fields,
    residual_size_penalty,
    total_derivative,
    validate_uniform_timestep,
    weighted_mse,
)


def test_coupled_derivative_preserves_cross_terms():
    # Both diagonal derivatives vanish; all acceleration comes from coupling.
    def field(x, _):
        return {"temperature": 2 * x["wind"], "wind": -3 * x["temperature"]}

    state = {"temperature": jnp.array(2.0), "wind": jnp.array(4.0)}
    result = jax.jit(lambda x: total_derivative(field, x))(state)
    np.testing.assert_allclose(result["temperature"], -12)
    np.testing.assert_allclose(result["wind"], -24)


def test_forcing_rate_and_clock_are_differentiated_but_clock_not_scored():
    def field(x, c):
        return {"x": 2 * x["x"] + 3 * c["sst"] + x["sim_time"], "sim_time": jnp.array(1.0)}

    state = {"x": jnp.array(2.0), "sim_time": jnp.array(5.0)}
    forcing = {"sst": jnp.array(4.0)}
    frozen = total_derivative(field, state, forcing)
    full = total_derivative(field, state, forcing, {"sst": jnp.array(5.0)})
    np.testing.assert_allclose(frozen["x"], 43)
    np.testing.assert_allclose(full["x"], 58)
    assert "sim_time" not in floating_dynamic_fields(full)


def test_acceleration_loss_has_correct_parameter_gradient():
    state = {"x": jnp.array(2.0), "y": jnp.array(3.0)}
    target = {"x": jnp.array(-4.0), "y": jnp.array(-6.0)}

    def objective(parameter):
        def field(x, _):
            return {"x": parameter * x["y"], "y": -x["x"]}
        return coupled_derivative_loss(field, state, target)

    # L = (4 + 9)/2 * (parameter - 2)^2.
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(jnp.array(1.0)), -13.0)
    np.testing.assert_allclose(objective(jnp.array(2.0)), 0.0)


def test_weighted_loss_scaling_quadrature_masks_and_nested_paths():
    prediction = {"temperature": jnp.array([[2.0, 4.0]]), "tracers": {"q": jnp.array([[3.0, 99.0]])}, "sim_time": jnp.array(99.0), "counter": 3}
    target = {"temperature": jnp.zeros((1, 2)), "tracers": {"q": jnp.zeros((1, 2))}, "sim_time": jnp.array(-99.0), "counter": 9}
    loss = weighted_mse(
        prediction, target,
        variable_scales={"temperature": 2.0, ("tracers", "q"): 3.0},
        variable_weights={"temperature": 2.0, "tracers/q": 1.0},
        spatial_weights=jnp.array([1.0, 3.0]),
        mask={"tracers/q": jnp.array([[True, False]])},
    )
    np.testing.assert_allclose(loss, (2 * (1 + 3 * 4) / 4 + 1) / 3)


def test_masked_nan_is_ignored_and_unmasked_nan_propagates():
    target = {"x": jnp.array([0.0, jnp.nan])}
    mask = jnp.array([True, False])
    objective = lambda x: weighted_mse({"x": x}, target, mask=mask)
    value, gradient = jax.value_and_grad(objective)(jnp.array([2.0, 7.0]))
    np.testing.assert_allclose(value, 4.0)
    np.testing.assert_allclose(gradient, [4.0, 0.0])
    assert jnp.isnan(weighted_mse({"x": jnp.ones(2)}, target))


def test_centered_difference_targets_uniform_units_and_time_axis():
    times = np.arange(5, dtype=float) * 0.5
    trajectory = {"x": jnp.asarray(times**2)[None, :], "sim_time": jnp.asarray(times)}
    targets = finite_difference_targets(trajectory, times=times, time_axis=1)
    np.testing.assert_allclose(targets.tendency["x"], (2 * times[1:-1])[None, :])
    np.testing.assert_allclose(targets.acceleration["x"], 2 * np.ones((1, 3)))
    assert "sim_time" not in targets.tendency
    with pytest.raises(ValueError, match="uniform"):
        validate_uniform_timestep(times=[0, 1, 3])
    with pytest.raises(ValueError, match="disagrees"):
        validate_uniform_timestep(dt=1, times=times)
    with pytest.raises(ValueError, match="positive"):
        finite_difference_targets({"x": jnp.ones(3)}, dt=0)


def test_residual_penalty_and_configuration_errors():
    np.testing.assert_allclose(residual_size_penalty({"x": jnp.array([2.0, 4.0])}, variable_scales={"x": 2.0}), 2.5)
    with pytest.raises(ValueError, match="positive"):
        weighted_mse({"x": jnp.ones(2)}, {"x": jnp.ones(2)}, variable_scales={"x": 0})
    with pytest.raises(ValueError, match="paths must match"):
        weighted_mse({"x": jnp.ones(2)}, {"y": jnp.ones(2)})


def test_dinosaur_state_paths_preserve_names_and_exclude_clock():
    from dinosaur import primitive_equations

    state = primitive_equations.State(
        vorticity=jnp.zeros((1, 2, 3)),
        divergence=jnp.zeros((1, 2, 3)),
        temperature_variation=jnp.ones((1, 2, 3)),
        log_surface_pressure=jnp.zeros((1, 2, 3)),
        tracers={"specific_humidity": jnp.zeros((1, 2, 3))},
        sim_time=jnp.array(100.0),
    )
    fields = floating_dynamic_fields(state)
    assert set(fields) == {"vorticity", "divergence", "temperature_variation", "log_surface_pressure", "tracers/specific_humidity"}
    # Nested dataclass state compares to flattened target without losing scale.
    expected = {key: jnp.zeros_like(value) for key, value in fields.items()}
    value = jax.jit(lambda x: weighted_mse(x, expected, variable_scales={"temperature_variation": 2.0}))(state)
    np.testing.assert_allclose(value, 0.25 / 5)
