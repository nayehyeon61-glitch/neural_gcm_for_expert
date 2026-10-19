"""Deterministic, separately parameterized heads for NeuralGCM learned physics.

The mapping changes only the learned nodal tendency map. All heads receive all
upstream features, so separating output blocks does not remove temperature/wind/
moisture coupling. These are *variable* experts, not identifiable radiation,
convection, or turbulence processes. The residual is a bounded deterministic
model discrepancy; it is not sampled noise.
"""

from collections.abc import Callable
import math
from typing import Any

from dinosaur import pytree_utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


_TEMPERATURE_FIELDS = frozenset({"temperature", "temperature_variation"})
_MOMENTUM_FIELDS = frozenset({"u", "v", "divergence", "vorticity"})
_PRESSURE_FIELDS = frozenset({"log_surface_pressure", "surface_pressure"})
_MOISTURE_FIELDS = frozenset({"specific_humidity"})


def _path_names(path: tuple[Any, ...]) -> tuple[str, ...]:
    """Extract semantic keys from JAX dictionary/dataclass pytree paths."""
    return tuple(
        str(getattr(key, "key", getattr(key, "name", getattr(key, "idx", key))))
        for key in path
    )


def _expert_for_path(path: tuple[Any, ...], include_pressure: bool) -> str:
    names = _path_names(path)
    field = names[-1] if names else ""
    # Upstream tracers (including cloud liquid/ice) retain their exact subtree.
    if "tracers" in names or field in _MOISTURE_FIELDS:
        return "moisture"
    if field in _TEMPERATURE_FIELDS:
        return "temperature"
    if field in _MOMENTUM_FIELDS:
        return "momentum"
    if field in _PRESSURE_FIELDS:
        if not include_pressure:
            raise ValueError(
                f"Active pressure output {'/'.join(names)!r} requires "
                "include_pressure=True. Mask it upstream to disable prediction."
            )
        return "pressure"
    raise ValueError(
        f"Unclassified active learned-physics output {'/'.join(names)!r}. "
        "Add an explicit expert assignment or mask the field upstream."
    )


@gin.register(denylist=["output_shapes"])
class DecomposedNodalMapping(hk.Module):
    """Drop-in nodal mapping with independent, coupled variable experts.

    ``output_shapes`` uses the exact upstream ``NodalMapping`` contract: array
    leaves containing ``(channels, longitude, latitude)`` and ``None`` for masked
    fields. A distinct ``tower_factory(output_size)`` instance is created for
    each active expert. Use the original checkpoint's tower factory to preserve
    its column-network architecture. Parameters must be trained for this new
    architecture; a pretrained monolithic tower is not implicitly duplicated.

    ``components(inputs)`` returns a dictionary of full output pytrees suitable
    for diagnostics and regularization before the upstream tendency transform.
    Each expert is zero outside its output block and retains every masked None.
    The residual spans all active outputs and is bounded elementwise by
    ``residual_scale`` in the network's pre-transform units.

    The factory must be deterministic at apply time (as the standard deterministic
    NeuralGCM ColumnTower is). This module has no noise, memory, or RNG calls.
    """

    def __init__(
        self,
        output_shapes: Any,
        tower_factory: Callable = gin.REQUIRED,
        residual_scale: float = 0.1,
        enable_residual: bool = True,
        include_pressure: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        self.output_shapes = output_shapes
        self.feature_axis = -3
        self.residual_scale = float(residual_scale)
        self.enable_residual = bool(enable_residual) and residual_scale > 0
        paths_and_shapes, self._tree_def = jax.tree_util.tree_flatten_with_path(
            output_shapes
        )
        self._indices: dict[str, list[int]] = {}
        self._spatial_shape: tuple[int, int] | None = None
        offset = 0
        for path, shape in paths_and_shapes:
            shape = np.asarray(shape)
            if shape.ndim != 1 or shape.size != 3:
                raise ValueError(
                    f"Expected a (channels, longitude, latitude) shape for "
                    f"{_path_names(path)!r}, got {shape!r}"
                )
            if not np.issubdtype(shape.dtype, np.integer) or np.any(shape <= 0):
                raise ValueError(f"Output dimensions must be positive integers: {shape!r}")
            spatial_shape = tuple(int(n) for n in shape[-2:])
            if self._spatial_shape is None:
                self._spatial_shape = spatial_shape
            elif self._spatial_shape != spatial_shape:
                raise ValueError("All nodal outputs must share longitude/latitude dimensions")
            expert = _expert_for_path(path, include_pressure)
            size = int(shape[self.feature_axis])
            self._indices.setdefault(expert, []).extend(range(offset, offset + size))
            offset += size
        self.output_size = offset

        self._towers: dict[str, Callable] = {}
        for expert in ("temperature", "momentum", "moisture", "pressure"):
            if expert in self._indices:
                # Explicit scopes ensure separately trainable parameters even
                # when all experts use the same upstream factory configuration.
                with hk.experimental.name_scope(f"expert_{expert}"):
                    self._towers[expert] = tower_factory(len(self._indices[expert]))
        self._residual_tower = None
        if self.enable_residual and self.output_size:
            with hk.experimental.name_scope("expert_residual"):
                self._residual_tower = tower_factory(self.output_size)

    def _component_arrays(self, inputs: Any) -> dict[str, jax.Array]:
        if not self.output_size:
            return {}
        array = pytree_utils.pack_pytree(inputs, self.feature_axis)
        if array.ndim != 3:
            raise ValueError(f"Expected inputs with ndim=3, got shape={array.shape}")
        if array.shape[-2:] != self._spatial_shape:
            raise ValueError(
                f"Input spatial shape {array.shape[-2:]} does not match "
                f"output spatial shape {self._spatial_shape}"
            )
        full_shape = (self.output_size, *array.shape[-2:])
        outputs = {}
        for expert, tower in self._towers.items():
            values = tower(array)
            indices = self._indices[expert]
            expected = (len(indices), *array.shape[-2:])
            if values.shape != expected:
                raise ValueError(
                    f"{expert} tower returned {values.shape}; expected {expected}"
                )
            outputs[expert] = jnp.zeros(full_shape, values.dtype).at[
                jnp.asarray(indices)
            ].set(values)
        if self._residual_tower is not None:
            raw = self._residual_tower(array)
            if raw.shape != full_shape:
                raise ValueError(
                    f"Residual tower returned {raw.shape}; expected {full_shape}"
                )
            outputs["residual"] = self.residual_scale * jnp.tanh(raw)
        return outputs

    def components(self, inputs: Any) -> dict[str, Any]:
        """Return additive component pytrees, including a scaled residual."""
        return {
            name: pytree_utils.unpack_to_pytree(
                values, self.output_shapes, self.feature_axis
            )
            for name, values in self._component_arrays(inputs).items()
        }

    def __call__(self, inputs: Any) -> Any:
        """Return the exact upstream output tree with all components summed."""
        components = self._component_arrays(inputs)
        if not components:
            return jax.tree_util.tree_unflatten(self._tree_def, [])
        arrays = iter(components.values())
        total = next(arrays)
        for array in arrays:
            total = total + array
        return pytree_utils.unpack_to_pytree(
            total, self.output_shapes, self.feature_axis
        )
