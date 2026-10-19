"""CPU integration smoke check using an explicitly converted official toy model.

Run ``python -m examples.smoke_test --checkpoint /tmp/expert-smoke.pkl``.
This is a code-path check, NOT a pretrained deterministic baseline or a forecast
skill evaluation. The source bundled checkpoint is stochastic; conversion
removes its random inputs and reinitializes incompatible/new layers. The one
available demo weather snapshot supplies a synthetic persistence target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from dinosaur import horizontal_interpolation, spherical_harmonic
import jax
import jax.numpy as jnp
import neuralgcm
from neuralgcm import demo
import numpy as np
import xarray as xr

from neuralgcm_expert.model import (
    convert_config, from_checkpoint, restore_checkpoint, save_checkpoint,
)


def make_tiny_expert_fixture(seed: int = 0):
    """Return a real official-solver toy model, demo inputs, and its provenance.

    Grid and network widths are reduced only for inexpensive integration tests.
    The physical core, sigma/pressure coordinate transforms, encoder, decoder,
    and IMEX integrator remain the official NeuralGCM implementations.
    """
    checkpoint = demo.load_checkpoint_tl63_stochastic()
    source_aux = xr.Dataset.from_dict(checkpoint["aux_ds_dict"])
    # The public API recognizes a fixed set of data grids; TL31 is its smallest.
    # Static feature validation also requires the model's nodal grid to match.
    grid = spherical_harmonic.Grid.TL31()
    regridder = horizontal_interpolation.ConservativeRegridder(
        spherical_harmonic.Grid.TL63(), grid
    )
    aux = demo._horizontal_regrid(regridder, source_aux)
    overrides = (
        "CoordinateSystem.horizontal = @GridTL31()",
        "GridTL31.spherical_harmonics_impl = @RealSphericalHarmonicsWithZeroImag",
        "smaller/GridWithWavenumbers.longitude_wavenumbers = 7",
        "smaller/GridWithWavenumbers.dealiasing = 'linear'",
        "LATENT_SIZE = 8",
        "LAYER_SIZE = 8",
        "NUM_BLOCKS = 1",
        "N_INNER_DYCORE_STEPS = 1",
        "get_model_specs.model_time_step = '10 minutes'",
    )
    checkpoint = dict(checkpoint, aux_ds_dict=aux.to_dict())
    # Constructing the structure does not run the stochastic source network.
    config = convert_config(checkpoint["model_config_str"], "experts", overrides)
    structure = neuralgcm.PressureLevelModel.from_checkpoint(
        dict(checkpoint, model_config_str=config)
    )
    data = demo.load_data(structure.data_coords)
    inputs, initial_forcings = structure.data_from_xarray(data.isel(time=0))
    model = from_checkpoint(
        checkpoint, mode="experts", sample_inputs=inputs,
        sample_forcings=initial_forcings, rng=jax.random.key(seed),
        gin_bindings=overrides, allow_stochastic_conversion=True,
    )
    return model, data, checkpoint


def run_smoke(checkpoint_path: str | Path | None = None, *, native_losses: bool = False) -> dict:
    """Encode → official rollout → loss → gradient → update → checkpoint reload."""
    from neuralgcm_expert.training import (
        LossConfig, make_train_step, prepare_native_targets, with_params,
    )

    model, data, template = make_tiny_expert_fixture()
    inputs, initial_forcings = model.data_from_xarray(data.isel(time=0))
    forcings = model.forcings_from_xarray(data)
    interval = np.timedelta64(10, "m")
    initial = model.encode(inputs, initial_forcings)
    assert initial.memory is None
    _, before = model.unroll(
        initial, forcings, steps=1, timedelta=interval, start_with_input=False
    )
    # A persistence target deliberately tests training without claiming that a
    # future observation exists in the bundled single-snapshot demonstration.
    batch = jax.tree_util.tree_map(jnp.asarray, {
        "inputs": inputs,
        "initial_forcings": initial_forcings,
        "forcings": forcings,
        "targets": {"temperature": inputs["temperature"][None, ...]},
    })
    native_scales = None
    loss_config = LossConfig(residual_parameter_weight=0)
    if native_losses:
        from neuralgcm_expert.losses import floating_dynamic_fields

        # These scales and persistence triplet only exercise derivative wiring;
        # production training must estimate scales/targets from training data.
        native_scales = {
            name: max(float(np.std(np.asarray(value))), 1.0)
            for name, value in floating_dynamic_fields(initial.state).items()
        }
        triplet = xr.concat([data.isel(time=0)] * 3, dim="time").assign_coords(
            time=data.time.values[0] + np.arange(3) * interval
        )
        batch["native"] = prepare_native_targets(
            model, triplet, interval, native_scales=native_scales
        )
        loss_config = LossConfig(
            tendency_weight=1e-5, acceleration_weight=1e-8,
            residual_parameter_weight=0,
        )
    _, opt_state, update = make_train_step(
        model, forecast_steps=1, output_interval=interval,
        variable_scales={"temperature": 10.0},
        loss_config=loss_config, learning_rate=1e-5,
        native_scales=native_scales,
    )
    next_params, _, metrics = update(model.params, opt_state, batch)
    metrics = {key: np.asarray(value).item() for key, value in metrics.items()}
    if not metrics["accepted"]:
        raise AssertionError(f"Official-solver gradient update was rejected: {metrics}")
    if native_losses and (metrics["tendency"] <= 0 or metrics["acceleration"] <= 0):
        raise AssertionError(f"Native derivative losses did not contribute: {metrics}")
    changes = {
        module: sum(float(jnp.sum(jnp.abs(next_params[module][name] - value)))
                    for name, value in values.items())
        for module, values in model.params.items()
    }
    expert_change = sum(value for key, value in changes.items() if "expert_" in key)
    frozen_change = sum(value for key, value in changes.items()
                        if "div_curl_neural_parameterization" not in key)
    if expert_change <= 0 or frozen_change != 0:
        raise AssertionError(f"Unexpected parameter changes: experts={expert_change}, frozen={frozen_change}")
    trained = with_params(model, next_params)
    if checkpoint_path is None:
        checkpoint_path = Path(tempfile.mkdtemp(prefix="neuralgcm-expert-smoke-")) / "model.pkl"
    saved = save_checkpoint(trained, checkpoint_path, template_checkpoint=template)
    reloaded = restore_checkpoint(saved)
    restored = reloaded.encode(inputs, initial_forcings)
    _, after_reload = reloaded.unroll(
        restored, forcings, steps=1, timedelta=interval, start_with_input=False
    )
    trained_state = trained.encode(inputs, initial_forcings)
    _, after = trained.unroll(
        trained_state, forcings, steps=1, timedelta=interval, start_with_input=False
    )
    for name in model.input_variables:
        if before[name].shape != after_reload[name].shape:
            raise AssertionError(f"Public output shape changed for {name}")
        np.testing.assert_allclose(after[name], after_reload[name], rtol=1e-6, atol=1e-6)
    return {
        "purpose": "converted stochastic toy integration smoke; not forecast skill evaluation",
        "grid": list(model.data_coords.horizontal.nodal_shape),
        "metrics": metrics,
        "native_losses_enabled": native_losses,
        "expert_parameter_change_l1": expert_change,
        "frozen_parameter_change_l1": frozen_change,
        "checkpoint": str(saved),
        "checkpoint_roundtrip": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--native-losses", action="store_true",
                        help="also differentiate nonzero native tendency and acceleration losses")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.checkpoint, native_losses=args.native_losses), indent=2))
