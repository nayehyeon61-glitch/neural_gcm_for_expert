"""A narrowly scoped bridge to the official NeuralGCM 1.2.2 inference API.

Only the learned-physics mapping is replaced in ``experts`` mode. Coordinate
transforms, physical equations, IMEX integration, encoders and decoders continue
to be implemented by NeuralGCM/Dinosaur. Removing native memory requires a
different encoder and input feature shapes, so converted models are experiments
that need training, never exact reinterpretations of pretrained weights.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import importlib.metadata
from pathlib import Path
import os
import pickle
import tempfile
from typing import Any, Literal
import warnings

import gin
import jax
import jax.numpy as jnp
import neuralgcm
from neuralgcm.legacy import gin_utils
import numpy as np

Mode = Literal["baseline", "memoryless", "experts"]
SUPPORTED_NEURALGCM_VERSION = "1.2.2"
_WITH_MEMORY = "DimensionalLearnedWeatherbenchToPrimitiveWithMemoryEncoder"
_WITHOUT_MEMORY = "DimensionalLearnedWeatherbenchToPrimitiveEncoder"
_EXCLUDED_FEATURES = (
    "MemoryVelocityAndValues", "MemoryPressureFeatures", "RandomnessFeatures",
)


def _check_version() -> None:
    version = importlib.metadata.version("neuralgcm")
    if version != SUPPORTED_NEURALGCM_VERSION:
        raise RuntimeError(
            f"This bridge is tested with neuralgcm=={SUPPORTED_NEURALGCM_VERSION}; "
            f"installed version is {version}. Install the pinned requirements."
        )


def _resolve_reference(value: Any) -> str:
    """Resolve Gin macros without constructing any Haiku modules."""
    result = repr(value)
    visited = set()
    while result.startswith("%"):
        if result in visited:
            raise ValueError(f"Cyclic Gin macro: {result}")
        visited.add(result)
        value = gin.query_parameter(result)
        result = repr(value)
    return result


def _configuration_bindings() -> list[tuple[str, str, dict[str, Any]]]:
    # Gin has no public binding-enumeration API. This single private dependency
    # is isolated here and covered by the pinned integration test.
    return [(scope, selector, dict(values))
            for (scope, selector), values in gin.config._CONFIG.items()]


def configuration_report(config: str) -> dict[str, Any]:
    """Inspect actual randomness bindings, rather than dormant Gaussian classes."""
    randomness = {}
    perturbations = {}
    with gin_utils.specific_config(config):
        for scope, selector, bindings in _configuration_bindings():
            prefix = f"{scope}/" if scope else ""
            for name, output in (("randomness_module", randomness),
                                 ("perturbation_module", perturbations)):
                if name in bindings:
                    output[f"{prefix}{selector}.{name}"] = _resolve_reference(bindings[name])
    deterministic_randomness = ("NoRandomField", "ZerosRandomField")
    stochastic = {
        key: value for key, value in randomness.items()
        if value.lstrip("@").split("/")[-1].split(".")[-1].removesuffix("()")
        not in deterministic_randomness
    }
    nontrivial_perturbations = {
        key: value for key, value in perturbations.items()
        if value.lstrip("@").split("/")[-1].split(".")[-1].removesuffix("()")
        != "NoPerturbation"
    }
    return {
        "randomness_bindings": randomness,
        "stochastic_bindings": stochastic,
        "perturbation_bindings": nontrivial_perturbations,
        "has_native_memory_encoder": _WITH_MEMORY in config,
    }


def _reference_bindings(value: Any, inherited_scope: str = "") -> tuple[str, str, dict]:
    """Resolve a configurable factory with the same scope rules as Gin calls."""
    reference = _resolve_reference(value)
    if not reference.startswith("@") or reference.endswith("()"):
        raise ValueError(f"Expected a configurable factory reference, got {reference}")
    *scopes, selector = reference[1:].split("/")
    # An explicitly scoped Gin reference sets the scope; an unscoped reference
    # inherits the scope of the enclosing configurable's constructor/method.
    scope = "/".join(scopes) if scopes else inherited_scope
    with gin.config_scope(scope.split("/") if scope else []):
        bindings = dict(gin.get_bindings(selector, resolve_references=False))
    return selector, scope, bindings


def _expert_mapping_bindings() -> list[str]:
    """Inherit the actual active learned mapping's factory and Gin scope.

    The supported route is the official legacy stochastic-capable modular step
    (configured deterministically here), DivCurlNeuralParameterization, and its
    NodalMapping. Alternate interfaces fail explicitly rather than receiving an
    unrelated default tower. No module is instantiated during inspection.
    """
    model_ref = gin.get_bindings("WhirlModel", resolve_references=False).get("model_cls")
    model_selector, model_scope, model_args = _reference_bindings(model_ref)
    if model_selector.split(".")[-1] != "StochasticModularStepModel":
        raise ValueError("Expert conversion requires the official StochasticModularStepModel interface")
    step_selector, step_scope, step_args = _reference_bindings(model_args.get("advance_module"), model_scope)
    if step_selector.split(".")[-1] != "StochasticPhysicsParameterizationStep":
        raise ValueError("Expert conversion requires StochasticPhysicsParameterizationStep; unsupported step wrapper")
    parameterization, physics_scope, physics_args = _reference_bindings(
        step_args.get("physics_parameterization_module"), step_scope
    )
    if parameterization.split(".")[-1] != "DivCurlNeuralParameterization":
        raise ValueError("Expert conversion requires DivCurlNeuralParameterization")
    mapping, mapping_scope, mapping_args = _reference_bindings(
        physics_args.get("nodal_mapping_module"), physics_scope
    )
    if mapping.split(".")[-1] not in {"NodalMapping", "DecomposedNodalMapping"}:
        raise ValueError(f"Unsupported learned mapping {mapping!r}; expected the official NodalMapping contract")
    if "tower_factory" not in mapping_args or mapping_args["tower_factory"] is gin.REQUIRED:
        raise ValueError(f"Source mapping {mapping!r} has no configured tower_factory")
    factory = _resolve_reference(mapping_args["tower_factory"])
    if not factory.startswith("@") or factory.endswith("()"):
        raise ValueError(f"Source tower_factory must be a configurable factory, got {factory}")
    new_mapping = "neuralgcm_expert.mappings.DecomposedNodalMapping"
    scoped_mapping = f"{mapping_scope}/{new_mapping}" if mapping_scope else new_mapping
    scoped_parameterization = f"{physics_scope}/{parameterization}" if physics_scope else parameterization
    return [
        f"{scoped_parameterization}.nodal_mapping_module = @{scoped_mapping}",
        f"{scoped_mapping}.tower_factory = {factory}",
    ]


def convert_config(config: str, mode: Mode,
                   gin_bindings: Sequence[str] = (), *, keep_memory: bool = False) -> str:
    """Return a no-memory/no-randomness configuration for supported legacy models.

    Excluding feature modules leaves normalization dictionaries intact. Gin
    retains inactive checkpoint definitions; their presence is not evidence of
    stochastic execution. Only the bound modules are materialized.

    ``keep_memory`` (experts only) keeps the official memory encoder and memory
    features, so only the learned-physics output mapping changes.
    """
    if mode not in ("baseline", "memoryless", "experts"):
        raise ValueError(f"Unknown mode: {mode!r}")
    if mode == "baseline":
        # Only value-preserving rematerialization may be added to the baseline.
        if any(not b.endswith("ColumnTower.checkpoint_tower = True") for b in gin_bindings):
            raise ValueError("baseline preserves the checkpoint exactly; only tower remat allowed")
        if gin_bindings:
            config += "\n# memory-only rematerialization\n" + "\n".join(gin_bindings)
        return config
    if keep_memory and mode != "experts":
        raise ValueError("keep_memory applies only to experts mode")
    if mode == "experts":
        from neuralgcm_expert import mappings as _mappings  # noqa: F401
    if not keep_memory:
        config = config.replace(_WITH_MEMORY, _WITHOUT_MEMORY)
    excluded_features = _EXCLUDED_FEATURES if not keep_memory else ("RandomnessFeatures",)
    additions = []
    with gin_utils.specific_config(config):
        bindings = _configuration_bindings()
        for scope, selector, values in bindings:
            prefix = f"{scope}/" if scope else ""
            key = prefix + selector
            if "randomness_module" in values:
                additions.append(f"{key}.randomness_module = @NoRandomField")
            if "perturbation_module" in values:
                additions.append(f"{key}.perturbation_module = @NoPerturbation")
            if selector.endswith(".CombinedFeatures"):
                excluded = tuple(values.get("feature_module_names_to_exclude", ()))
                excluded = tuple(dict.fromkeys(excluded + excluded_features))
                additions.append(f"{key}.feature_module_names_to_exclude = {excluded!r}")
        # Include defaults even if these kwargs were absent in the source Gin.
        encoder = _WITH_MEMORY if keep_memory else _WITHOUT_MEMORY
        additions.extend([
            f"{encoder}.randomness_module = @NoRandomField",
            f"{encoder}.perturbation_module = @NoPerturbation",
            "StochasticPhysicsParameterizationStep.randomness_module = @NoRandomField",
            "StochasticPhysicsParameterizationStep.perturbation_module = @NoPerturbation",
        ])
        if mode == "experts":
            additions.extend(_expert_mapping_bindings())
    result = config + "\n# neuralgcm_expert conversion\n" + "\n".join(additions)
    if gin_bindings:
        result += "\n# explicit experiment overrides\n" + "\n".join(gin_bindings)
    report = configuration_report(result)
    if report["stochastic_bindings"] or report["perturbation_bindings"]:
        raise ValueError("Experiment overrides re-enabled randomness/perturbations")
    if report["has_native_memory_encoder"] and not keep_memory:
        raise ValueError("Experiment overrides re-enabled a memory encoder")
    return result


def transfer_matching_params(initialized: Mapping[str, Mapping[str, Any]],
                             source: Mapping[str, Mapping[str, Any]],
                             module_aliases: Mapping[str, str] | None = None,
                             ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Copy compatible paths, optionally using audited encoder-scope aliases.

    Aliases must name a whole module path, not approximate substring matches.
    The optional bridge-generated alias strips the *state* child encoder of a
    WithMemory wrapper; it never copies the discarded memory child's weights.
    """
    module_aliases = dict(module_aliases or {})
    result = {}
    transferred, initialized_paths, mismatches = [], [], []
    used_source_paths, used_aliases = set(), {}
    for module, parameters in initialized.items():
        result[module] = {}
        source_module = module if module in source else module_aliases.get(module, module)
        for name, value in parameters.items():
            path = f"{module}/{name}"
            old = source.get(source_module, {}).get(name)
            if old is not None and np.shape(old) == np.shape(value) and np.asarray(old).dtype == np.asarray(value).dtype:
                result[module][name] = jnp.asarray(old)
                transferred.append(path)
                source_path = f"{source_module}/{name}"
                used_source_paths.add(source_path)
                if source_module != module:
                    used_aliases[path] = source_path
            else:
                result[module][name] = value
                initialized_paths.append(path)
                if old is not None:
                    mismatches.append({"path": path, "source_shape": list(np.shape(old)),
                                       "target_shape": list(np.shape(value))})
    source_paths = {f"{m}/{n}" for m, p in source.items() for n in p}
    return result, {
        "transferred": transferred,
        "initialized": initialized_paths,
        "shape_or_dtype_mismatches": mismatches,
        "encoder_state_aliases": used_aliases,
        "unused_source": sorted(source_paths - used_source_paths),
        "exact_pretrained_equivalence": False,
    }


def _state_encoder_aliases(config: str, initialized: Mapping, source: Mapping) -> dict[str, str]:
    """Match the surviving state encoder of the official WithMemory wrapper.

    Upstream's wrapper creates memory_encoder first, state_encoder second. The
    latter is therefore named learned_weatherbench_to_primitive_encoder_1 (or
    <explicit_name>_1). Its computation matches the direct dimensional encoder
    after nondimensionalization, so an exact leaf shape/dtype check is valid.
    """
    if _WITH_MEMORY not in config:
        return {}
    names = {None}
    with gin_utils.specific_config(config):
        for _, selector, values in _configuration_bindings():
            if selector.endswith("." + _WITH_MEMORY):
                names.add(values.get("name"))
    pairs = []
    for name in names:
        old_outer = name or "dimensional_learned_weatherbench_to_primitive_with_memory_encoder"
        new_outer = name or "dimensional_learned_weatherbench_to_primitive_encoder"
        child = (name or "learned_weatherbench_to_primitive_encoder") + "_1"
        pairs.append((new_outer, old_outer, child))
    aliases = {}
    for module in initialized:
        if module in source:
            continue
        # Prefixing a slash supports an encoder at the root as well as nested
        # model constructors without matching similar module-name substrings.
        candidates = set()
        for new_outer, old_outer, child in pairs:
            marker = f"/{new_outer}/"
            if marker in "/" + module + "/":
                renamed = ("/" + module + "/").replace(
                    marker, f"/{old_outer}/~/{child}/", 1
                ).strip("/")
                if renamed in source:
                    candidates.add(renamed)
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous state encoder weight aliases for {module}")
        if candidates:
            aliases[module] = candidates.pop()
    return aliases


def from_checkpoint(
    checkpoint: Mapping[str, Any], mode: Mode = "baseline", *,
    sample_inputs: Mapping[str, Any] | None = None,
    sample_forcings: Mapping[str, Any] | None = None,
    rng: jax.Array | None = None,
    gin_bindings: Sequence[str] = (),
    allow_stochastic_conversion: bool = False,
    keep_memory: bool = False,
) -> neuralgcm.PressureLevelModel:
    """Load an untouched baseline or initialize a modified learned-physics model.

    Inputs/forcings are one time slice in the official PressureLevelModel format
    (including sim_time and a length-one level axis on surface forcings).
    Conversion initializes the *entire* new encode/advance/decode graph before
    copying compatible parameters. New heads and incompatible input layers need
    training. Baseline mode never changes parameters or requires samples.

    Stochastic checkpoints are rejected by default for all modes. The explicit
    ``allow_stochastic_conversion`` switch exists for controlled conversion and
    integration fixtures, and can never be used for an untouched baseline.
    """
    _check_version()
    required = {"model_config_str", "aux_ds_dict", "params"}
    if missing := required - set(checkpoint):
        raise ValueError(f"Checkpoint is missing official fields: {sorted(missing)}")
    source_report = configuration_report(checkpoint["model_config_str"])
    stochastic = bool(source_report["stochastic_bindings"] or source_report["perturbation_bindings"])
    if stochastic and (mode == "baseline" or not allow_stochastic_conversion):
        raise ValueError(
            "This checkpoint enables stochastic randomness/perturbations. Use an "
            "official deterministic checkpoint. Controlled reinitialization can "
            "explicitly set allow_stochastic_conversion=True in memoryless/experts mode."
        )
    config = convert_config(checkpoint["model_config_str"], mode, gin_bindings,
                            keep_memory=keep_memory)
    converted = dict(checkpoint, model_config_str=config)
    model = neuralgcm.PressureLevelModel.from_checkpoint(converted)
    metadata = {
        "mode": mode,
        "neuralgcm_version": SUPPORTED_NEURALGCM_VERSION,
        "source_was_stochastic": stochastic,
        "source_native_memory": source_report["has_native_memory_encoder"],
        "requires_training": mode != "baseline",
        "native_memory_kept": mode == "baseline" or keep_memory,
        "original_checkpoint_config": checkpoint["model_config_str"],
    }
    if mode != "baseline":
        if sample_inputs is None or sample_forcings is None or rng is None:
            raise ValueError("Converted modes require sample_inputs, sample_forcings and rng for full initialization")
        inputs = model._to_abbreviated_names_and_tracers(dict(sample_inputs))
        trajectory = jax.tree.map(lambda x: jnp.expand_dims(x, 0), inputs)
        forcings = model._squeeze_level_from_forcings(dict(sample_forcings))
        forcing_data = jax.tree.map(lambda x: jnp.expand_dims(x, 0), forcings)
        with gin_utils.specific_config(model.gin_config):
            initialized = model._structure.init_params(rng, trajectory, forcing_data)
        aliases = _state_encoder_aliases(checkpoint["model_config_str"], initialized, checkpoint["params"])
        params, transfer_report = transfer_matching_params(initialized, checkpoint["params"], aliases)
        model = neuralgcm.PressureLevelModel(model._structure, params, model.gin_config)
        metadata["parameter_transfer"] = transfer_report
        if stochastic:
            warnings.warn(
                "Converting a stochastic checkpoint: random features removed and "
                "incompatible layers reinitialized. This is not a pretrained "
                "deterministic NeuralGCM baseline; retraining is required.",
                UserWarning, stacklevel=2,
            )
    else:
        metadata["parameter_transfer"] = {"exact_pretrained_equivalence": True}
    metadata["total_parameter_count"] = sum(int(np.size(x)) for x in jax.tree.leaves(model.params))
    model.expert_metadata = metadata
    model._checkpoint_aux_ds_dict = checkpoint["aux_ds_dict"]
    return model


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read an official pickle checkpoint. Only load trusted pickle files."""
    with Path(path).open("rb") as stream:
        checkpoint = pickle.load(stream)
    required = {"model_config_str", "aux_ds_dict", "params"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Not an official NeuralGCM checkpoint dictionary")
    return checkpoint


def restore_checkpoint(checkpoint: Mapping[str, Any] | str | Path) -> neuralgcm.PressureLevelModel:
    """Restore trained weights without conversion or reinitialization.

    Custom mapping registration is explicit and checkpoint provenance is
    restored on the returned, otherwise unmodified PressureLevelModel.
    """
    _check_version()
    from neuralgcm_expert import mappings as _mappings  # noqa: F401
    if isinstance(checkpoint, (str, Path)):
        checkpoint = load_checkpoint(checkpoint)
    report = configuration_report(checkpoint["model_config_str"])
    if report["stochastic_bindings"] or report["perturbation_bindings"]:
        raise ValueError("restore_checkpoint only accepts deterministic experiment checkpoints")
    model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)
    model.expert_metadata = copy.deepcopy(checkpoint.get("expert_metadata", {
        "mode": "baseline", "requires_training": False,
        "neuralgcm_version": SUPPORTED_NEURALGCM_VERSION,
    }))
    model._checkpoint_aux_ds_dict = checkpoint["aux_ds_dict"]
    return model


def save_checkpoint(model: neuralgcm.PressureLevelModel, path: str | Path,
                    template_checkpoint: Mapping[str, Any] | None = None,
                    *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Atomically save official-schema weights, plus optional provenance metadata.

    Reload custom mappings after importing ``neuralgcm_expert.mappings``; the
    checkpoint uses regular Gin references, never serialized function closures.
    """
    aux = (template_checkpoint["aux_ds_dict"] if template_checkpoint is not None
           else getattr(model, "_checkpoint_aux_ds_dict", None))
    if aux is None:
        raise ValueError("Provide template_checkpoint to preserve official auxiliary dataset metadata")
    provenance = copy.deepcopy(getattr(model, "expert_metadata", {}))
    provenance.update(dict(metadata or {}))
    provenance["total_parameter_count"] = sum(int(np.size(x)) for x in jax.tree.leaves(model.params))
    checkpoint = {
        "model_config_str": model.gin_config,
        "aux_ds_dict": aux,
        "params": jax.tree.map(lambda x: np.asarray(x), model.params),
        "expert_metadata": provenance,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path
