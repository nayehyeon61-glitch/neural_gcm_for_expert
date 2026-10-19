"""Compatibility, configuration and checkpoint safety tests."""
import gin
import haiku as hk
import jax
import jax.numpy as jnp
import neuralgcm
import neuralgcm.demo
from neuralgcm.legacy import gin_utils
import numpy as np
import pytest

from neuralgcm_expert.model import (
    _state_encoder_aliases,
    configuration_report, convert_config, from_checkpoint, load_checkpoint,
    restore_checkpoint, save_checkpoint, transfer_matching_params,
)


@pytest.fixture(scope="module")
def official_checkpoint():
    return neuralgcm.demo.load_checkpoint_tl63_stochastic()


def test_stochastic_checkpoint_is_rejected_by_default(official_checkpoint):
    for mode in ("baseline", "memoryless", "experts"):
        with pytest.raises(ValueError, match="stochastic"):
            from_checkpoint(official_checkpoint, mode)
    with pytest.raises(ValueError, match="stochastic"):
        from_checkpoint(official_checkpoint, "baseline", allow_stochastic_conversion=True)


def test_convert_config_removes_native_memory_and_random_features(official_checkpoint):
    source = official_checkpoint["model_config_str"].replace(
        "DimensionalLearnedWeatherbenchToPrimitiveEncoder",
        "DimensionalLearnedWeatherbenchToPrimitiveWithMemoryEncoder",
    )
    result = convert_config(source, "memoryless")
    report = configuration_report(result)
    assert not report["has_native_memory_encoder"]
    assert not report["stochastic_bindings"]
    with gin_utils.specific_config(result):
        excluded = gin.query_parameter("advance/CombinedFeatures.feature_module_names_to_exclude")
        assert "MemoryVelocityAndValues" in excluded
        assert "MemoryPressureFeatures" in excluded
        assert "RandomnessFeatures" in excluded
        assert repr(gin.query_parameter("StochasticPhysicsParameterizationStep.randomness_module")) == "@NoRandomField"
        assert repr(gin.query_parameter("StochasticPhysicsParameterizationStep.corrector_module")) == "%CORRECTOR_MODULE"


def test_experts_changes_only_learned_mapping_and_accepts_ablation(official_checkpoint):
    result = convert_config(official_checkpoint["model_config_str"], "experts", [
        "DecomposedNodalMapping.enable_residual = False",
    ])
    with gin_utils.specific_config(result):
        assert "DecomposedNodalMapping" in repr(gin.query_parameter("DivCurlNeuralParameterization.nodal_mapping_module"))
        assert gin.query_parameter("DecomposedNodalMapping.enable_residual") is False
        assert repr(gin.query_parameter("DycoreWithPhysicsCorrector.time_integrator")) == "%DYCORE_INTEGRATOR"
        assert repr(gin.query_parameter("DecomposedNodalMapping.tower_factory")) == "@EpdTower"


@pytest.mark.parametrize("scoped_parameterization", [False, True])
def test_experts_inherits_active_scoped_tower_without_manual_override(official_checkpoint, scoped_parameterization):
    source = official_checkpoint["model_config_str"]
    if scoped_parameterization:
        # Unscoped NodalMapping inherits the scope of its parameterization.
        source += "\nStochasticPhysicsParameterizationStep.physics_parameterization_module = @advance/DivCurlNeuralParameterization"
    else:
        # An explicitly scoped mapping sets its own scope.
        source += "\nDivCurlNeuralParameterization.nodal_mapping_module = @advance/NodalMapping"
    source += "\nadvance/NodalMapping.tower_factory = @process/ColumnTower"
    converted = convert_config(source, "experts")
    with gin_utils.specific_config(converted):
        prefix = "advance/" if scoped_parameterization else ""
        assert repr(gin.query_parameter(prefix + "DivCurlNeuralParameterization.nodal_mapping_module")) == "@advance/neuralgcm_expert.mappings.DecomposedNodalMapping"
        assert repr(gin.query_parameter("advance/DecomposedNodalMapping.tower_factory")) == "@process/ColumnTower"


def test_experts_rejects_unsupported_source_mapping(official_checkpoint):
    source = official_checkpoint["model_config_str"] + "\nDivCurlNeuralParameterization.nodal_mapping_module = @ColumnTower"
    with pytest.raises(ValueError, match="Unsupported learned mapping"):
        convert_config(source, "experts")


def test_inherited_tower_initializes_without_hidden_required_bindings(official_checkpoint):
    converted = convert_config(official_checkpoint["model_config_str"], "experts")
    shapes = {"temperature_variation": np.array([2, 4, 2]),
              "u": np.array([2, 4, 2]), "v": np.array([2, 4, 2]),
              "tracers": {"specific_humidity": np.array([2, 4, 2])},
              "log_surface_pressure": None, "sim_time": None}
    with gin_utils.specific_config(converted):
        factory = gin.get_configurable("DecomposedNodalMapping")
        transform = hk.transform(lambda features: factory(shapes)(features))
        features = {"all_variables": jnp.ones((5, 4, 2))}
        params = transform.init(jax.random.key(42), features)
        output = transform.apply(params, None, features)
    assert output["temperature_variation"].shape == (2, 4, 2)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(output))


def test_overrides_cannot_reenable_randomness(official_checkpoint):
    with pytest.raises(ValueError, match="re-enabled"):
        convert_config(official_checkpoint["model_config_str"], "memoryless", [
            "StochasticPhysicsParameterizationStep.randomness_module = @BatchGaussianRandomFieldModule",
        ])


def test_transfer_never_broadcasts_or_renames_weights():
    new = {"same": {"w": jnp.zeros((2, 3)), "b": jnp.zeros(3)},
           "new": {"w": jnp.zeros((1, 2))}}
    old = {"same": {"w": jnp.ones((2, 3)), "b": jnp.ones(4)},
           "old": {"w": jnp.ones((1, 2))}}
    transferred, report = transfer_matching_params(new, old)
    np.testing.assert_array_equal(transferred["same"]["w"], 1)
    np.testing.assert_array_equal(transferred["same"]["b"], 0)
    np.testing.assert_array_equal(transferred["new"]["w"], 0)
    assert report["transferred"] == ["same/w"]
    assert report["exact_pretrained_equivalence"] is False
    assert report["shape_or_dtype_mismatches"][0]["path"] == "same/b"


def test_memory_wrapper_transfers_state_encoder_never_memory_encoder(official_checkpoint):
    config = official_checkpoint["model_config_str"].replace(
        "DimensionalLearnedWeatherbenchToPrimitiveEncoder",
        "DimensionalLearnedWeatherbenchToPrimitiveWithMemoryEncoder",
    )
    prefix = "stochastic_modular_step_model/~/"
    new = prefix + "dimensional_learned_weatherbench_to_primitive_encoder/~/linear"
    old = prefix + "dimensional_learned_weatherbench_to_primitive_with_memory_encoder/~/"
    memory = old + "learned_weatherbench_to_primitive_encoder/~/linear"
    state = old + "learned_weatherbench_to_primitive_encoder_1/~/linear"
    initialized = {new: {"w": jnp.zeros((2, 3))}}
    source = {memory: {"w": jnp.full((2, 3), 99.0)}, state: {"w": jnp.ones((2, 3))}}
    aliases = _state_encoder_aliases(config, initialized, source)
    assert aliases == {new: state}
    transferred, report = transfer_matching_params(initialized, source, aliases)
    np.testing.assert_array_equal(transferred[new]["w"], 1)
    assert report["encoder_state_aliases"] == {new + "/w": state + "/w"}
    assert memory + "/w" in report["unused_source"]


def test_baseline_is_actual_official_model_and_portable_checkpoint(official_checkpoint, tmp_path):
    # Configuration-only unit fixture: this intentionally does not assert these
    # stochastic source weights are compatible with the converted architecture.
    checkpoint = dict(official_checkpoint, model_config_str=convert_config(
        official_checkpoint["model_config_str"], "memoryless"))
    model = from_checkpoint(checkpoint, "baseline")
    assert type(model) is neuralgcm.PressureLevelModel
    assert model.params is checkpoint["params"]
    assert model.expert_metadata["parameter_transfer"]["exact_pretrained_equivalence"]
    path = save_checkpoint(model, tmp_path / "checkpoint.pkl")
    saved = load_checkpoint(path)
    restored = restore_checkpoint(path)
    assert {"params", "aux_ds_dict", "model_config_str"}.issubset(saved)
    assert restored.input_variables == model.input_variables
    assert restored.forcing_variables == model.forcing_variables
    assert restored.timestep == model.timestep
    assert restored.expert_metadata["mode"] == "baseline"
    for module, params in model.params.items():
        for name, value in params.items():
            np.testing.assert_array_equal(restored.params[module][name], value)


def test_modified_model_requires_initialization_samples(official_checkpoint):
    with pytest.raises(ValueError, match="sample_inputs"):
        from_checkpoint(official_checkpoint, "memoryless", allow_stochastic_conversion=True)
