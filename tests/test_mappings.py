"""Contract and coupling checks against the official NeuralGCM column tower."""

import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import towers
import numpy as np
import pytest

from neuralgcm_expert.mappings import DecomposedNodalMapping


def output_shapes(include_pressure=False):
    return {
        "temperature_variation": np.array((2, 4, 3)),
        "u": np.array((2, 4, 3)),
        "v": np.array((2, 4, 3)),
        "tracers": {"specific_humidity": np.array((2, 4, 3)), "unused": None},
        "log_surface_pressure": np.array((1, 4, 3)) if include_pressure else None,
        "sim_time": None,
    }


def inputs():
    return {
        "temperature": jnp.ones((2, 4, 3)),
        "wind": jnp.full((2, 4, 3), 0.25),
        "humidity": jnp.full((2, 4, 3), 0.01),
        "forcing": jnp.full((1, 4, 3), 0.1),
    }


def tower_factory(size):
    return towers.ColumnTower(
        size,
        column_net_factory=lambda n: hk.nets.MLP([7, n], activation=jnp.tanh),
    )


def transform(shapes=None, **kwargs):
    shapes = output_shapes() if shapes is None else shapes

    def fn(features):
        mapping = DecomposedNodalMapping(shapes, tower_factory=tower_factory, **kwargs)
        return mapping(features), mapping.components(features)

    return hk.without_apply_rng(hk.transform(fn))


def test_exact_pytree_shapes_none_masks_and_sum_of_independent_components():
    shapes = output_shapes(include_pressure=True)
    model = transform(shapes)
    params = model.init(jax.random.PRNGKey(0), inputs())
    total, components = model.apply(params, inputs())
    assert set(components) == {"temperature", "momentum", "moisture", "pressure", "residual"}
    assert jax.tree_util.tree_structure(total) == jax.tree_util.tree_structure(shapes)
    jax.tree_util.tree_map(lambda x, s: np.testing.assert_equal(x.shape, tuple(s)), total, shapes)
    summed = jax.tree_util.tree_map(lambda *xs: sum(xs), *components.values())
    jax.tree_util.tree_map(np.testing.assert_allclose, total, summed)
    assert total["sim_time"] is None
    assert total["tracers"]["unused"] is None
    for name in ("temperature", "momentum", "moisture", "pressure", "residual"):
        assert any(f"expert_{name}/" in key for key in params)
    np.testing.assert_array_equal(components["temperature"]["u"], 0)
    np.testing.assert_array_equal(components["momentum"]["temperature_variation"], 0)
    np.testing.assert_array_equal(components["moisture"]["v"], 0)


def test_residual_is_bounded_and_deterministic_without_apply_rng():
    model = transform(residual_scale=0.03)
    params = model.init(jax.random.PRNGKey(1), inputs())
    out1, components = model.apply(params, inputs())
    out2, _ = jax.jit(model.apply)(params, inputs())
    jax.tree_util.tree_map(lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7), out1, out2)
    for leaf in jax.tree_util.tree_leaves(components["residual"]):
        assert float(jnp.max(jnp.abs(leaf))) <= 0.030001


@pytest.mark.parametrize("kwargs", [{"enable_residual": False}, {"residual_scale": 0.0}])
def test_residual_can_be_disabled_without_allocating_unused_parameters(kwargs):
    model = transform(**kwargs)
    params = model.init(jax.random.PRNGKey(2), inputs())
    _, components = model.apply(params, inputs())
    assert "residual" not in components
    assert not any("expert_residual" in key for key in params)


def test_each_expert_uses_full_input_and_temperature_depends_on_wind_and_humidity():
    # Positive linear weights make a zero cross-gradient unambiguously an input
    # routing bug, independent of random saturation or cancellation.
    def all_feature_tower(size):
        return towers.ColumnTower(
            size,
            column_net_factory=lambda n: hk.Linear(
                n, w_init=hk.initializers.Constant(1.0), with_bias=False
            ),
        )

    def fn(features):
        return DecomposedNodalMapping(
            output_shapes(), tower_factory=all_feature_tower, enable_residual=False
        ).components(features)

    model = hk.without_apply_rng(hk.transform(fn))
    params = model.init(jax.random.PRNGKey(3), inputs())
    for value in params.values():
        assert value["w"].shape[0] == 7  # All input channels reach every expert.
    gradients = jax.grad(
        lambda features: model.apply(params, features)["temperature"]["temperature_variation"].sum()
    )(inputs())
    for feature in ("wind", "humidity", "forcing"):
        assert bool(jnp.all(gradients[feature] != 0))


def test_pressure_is_optional_but_active_outputs_are_never_silently_dropped():
    model = transform(include_pressure=False)
    params = model.init(jax.random.PRNGKey(4), inputs())
    total, components = model.apply(params, inputs())
    assert total["log_surface_pressure"] is None
    assert "pressure" not in components
    with pytest.raises(ValueError, match="requires include_pressure=True"):
        transform(output_shapes(include_pressure=True), include_pressure=False).init(
            jax.random.PRNGKey(4), inputs()
        )


def test_unclassified_active_output_fails_instead_of_losing_a_tendency():
    shapes = output_shapes()
    shapes["ocean_temperature"] = np.array((1, 4, 3))
    with pytest.raises(ValueError, match="Unclassified active.*ocean_temperature"):
        transform(shapes).init(jax.random.PRNGKey(5), inputs())


def test_all_masked_outputs_keep_the_original_tree_without_parameters():
    shapes = {"temperature_variation": None, "tracers": {"q": None}, "sim_time": None}
    model = transform(shapes)
    params = model.init(jax.random.PRNGKey(6), inputs())
    total, components = model.apply(params, inputs())
    assert params == {}
    assert total == shapes
    assert components == {}


@pytest.mark.parametrize("scale", [-0.1, float("nan"), float("inf")])
def test_invalid_residual_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        transform(residual_scale=scale).init(jax.random.PRNGKey(7), inputs())
