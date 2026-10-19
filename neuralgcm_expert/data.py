"""Strict ERA5 data windows using the unmodified NeuralGCM public I/O API.

Inputs must already be on the checkpoint's pressure levels and horizontal grid.
Regridding, filling missing SST/sea-ice values, and unit conversions belong in an
explicit preprocessing step (see NeuralGCM's data-preparation notebook).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def open_dataset(path: str | Path) -> xr.Dataset:
    """Open a NetCDF file or Zarr store without renaming/regridding its data."""
    path = str(path)
    if path.rstrip("/").endswith(".zarr") or Path(path).is_dir():
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _require_time(dataset: xr.Dataset) -> np.ndarray:
    if "time" not in dataset.dims or "time" not in dataset.coords:
        raise ValueError("dataset must have a one-dimensional datetime64 time coordinate")
    time = dataset.time.values
    if time.ndim != 1 or not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError("time must be a one-dimensional datetime64 coordinate")
    if time.size == 0 or np.isnat(time).any():
        raise ValueError("time must be nonempty and contain no NaT values")
    if np.any(np.diff(time) <= np.timedelta64(0, "ns")):
        raise ValueError("time must be unique and strictly increasing")
    return time


def validate_coordinates(dataset: xr.Dataset, model: Any) -> None:
    """Reject another grid, reversed axes, or mismatched pressure levels.

    The 1e-5 degree/hPa tolerance accommodates coordinate serialization rounding;
    it does not permit interpolation, sorting, or dropping grid cells.
    """
    expected = {
        "longitude": np.rad2deg(model.data_coords.horizontal.longitudes),
        "latitude": np.rad2deg(model.data_coords.horizontal.latitudes),
        "level": np.asarray(model.data_coords.vertical.centers),
    }
    for name, desired in expected.items():
        if name not in dataset.coords:
            raise ValueError(f"missing {name} coordinate; use official full ERA5 coordinate names")
        actual = np.asarray(dataset[name].values)
        if dataset[name].dims != (name,):
            raise ValueError(f"{name} must be a one-dimensional coordinate")
        if actual.shape != desired.shape or not np.allclose(
            actual, desired, rtol=0, atol=1e-5, equal_nan=False
        ):
            raise ValueError(
                f"{name} coordinate mismatch with checkpoint; explicitly prepare the "
                "data on model.data_coords before training/evaluation"
            )


def _check_finite(values: dict, label: str) -> None:
    for name, value in values.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(
                f"{label}.{name} contains non-finite values; explicitly preprocess "
                "missing data following NeuralGCM data preparation"
            )


def output_timedelta(model: Any, output_interval: Any) -> np.timedelta64:
    """Validate the requested save interval against the actual dycore timestep."""
    duration = pd.Timedelta(output_interval)
    internal = pd.Timedelta(model.timestep)
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("output_interval must be positive")
    ratio = duration / internal
    if not np.isfinite(ratio) or abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
        raise ValueError("output_interval must be an integer multiple of model.timestep")
    return duration.to_timedelta64()


@dataclass(frozen=True)
class TrainingWindow:
    """One initialization and future targets, in official pressure-level format."""

    inputs: dict
    initial_forcings: dict
    forcings: dict
    targets: dict
    target_dataset: xr.Dataset
    initial_time: np.datetime64
    valid_times: np.ndarray
    output_interval: np.timedelta64
    forcing_mode: str


def make_training_window(
    model: Any,
    dataset: xr.Dataset,
    forecast_steps: int,
    output_interval: Any = "6h",
    *,
    start_index: int = 0,
    forcing_mode: str = "persistence",
) -> TrainingWindow:
    """Prepare a complete rollout window; targets start at the first future lead.

    ``persistence`` repeats initial SST/sea ice, as in the official inference demo.
    ``dynamic`` supplies observed forcing values through the forecast horizon;
    this is a prescribed-boundary hindcast, not an operational forecast protocol.
    The model uses its official nearest-time forcing interpolation in either case.
    """
    if isinstance(forecast_steps, bool) or not isinstance(forecast_steps, (int, np.integer)) or forecast_steps < 1:
        raise ValueError("forecast_steps must be a positive integer")
    if forcing_mode not in {"persistence", "dynamic"}:
        raise ValueError("forcing_mode must be 'persistence' or 'dynamic'")
    times = _require_time(dataset)
    if isinstance(start_index, bool) or not isinstance(start_index, (int, np.integer)) or not 0 <= start_index < len(times):
        raise ValueError("start_index must identify an existing initialization time")
    validate_coordinates(dataset, model)
    interval = output_timedelta(model, output_interval)
    initial_time = times[start_index]
    valid_times = initial_time + np.arange(1, forecast_steps + 1) * interval
    missing = valid_times[~np.isin(valid_times, times)]
    if missing.size:
        raise ValueError(f"missing exact forecast target times: {missing.tolist()}")
    initial = dataset.isel(time=start_index)
    target = dataset.sel(time=valid_times)[model.input_variables]
    forcing_data = (
        dataset.isel(time=slice(start_index, start_index + 1))
        if forcing_mode == "persistence"
        else dataset.sel(time=slice(initial_time, valid_times[-1]))
    )
    # These public methods also validate required names/dimensions and attach
    # NeuralGCM's nondimensional sim_time. Do not construct those dictionaries by hand.
    inputs = model.inputs_from_xarray(initial)
    initial_forcings = model.forcings_from_xarray(initial)
    forcings = model.forcings_from_xarray(forcing_data)
    targets = model.inputs_from_xarray(target)
    for label, values in (("inputs", inputs), ("initial_forcings", initial_forcings),
                          ("forcings", forcings), ("targets", targets)):
        _check_finite(values, label)
    return TrainingWindow(inputs, initial_forcings, forcings, targets, target,
                          initial_time, valid_times, interval, forcing_mode)
