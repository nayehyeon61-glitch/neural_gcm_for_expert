"""Native-grid deterministic evaluation with NeuralGCM's spherical quadrature.

The RMSE reduction is the one in upstream ``reference_code/metrics.py:RMSE``
and ``metrics_util.py:nodal_surface_mean``: sqrt(integrate(error**2)/(4*pi*R**2)).
Those research-reference modules are not included in the PyPI distribution, so
we call the very same Dinosaur grid integration directly. This preserves Gauss
quadrature; cosine-latitude weights are not a substitute on a Gaussian grid.

This is a reproducible per-initialization evaluation, not a claim to reproduce
the paper's complete WeatherBench2 protocol (dates, aggregation and grid).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from .data import _require_time, make_training_window, output_timedelta, validate_coordinates


def quadrature_weights(horizontal_grid: Any) -> np.ndarray:
    """Normalized (longitude, latitude) weights from the official grid basis."""
    weights = np.asarray(horizontal_grid.quadrature_weights, dtype=np.float64)
    weights = np.broadcast_to(weights, horizontal_grid.nodal_shape)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid horizontal quadrature weights")
    return weights / weights.sum()


def spherical_mean(values: Any, data_coords: Any):
    """Official nodal surface mean, keeping all nonspatial leading axes."""
    values = jnp.asarray(values)
    grid = data_coords.horizontal
    if values.shape[-2:] != grid.nodal_shape:
        raise ValueError(f"spatial shape {values.shape[-2:]} does not match {grid.nodal_shape}")
    if grid.radius is None or grid.radius <= 0:
        raise ValueError("the checkpoint horizontal grid must have a positive radius")
    return grid.integrate(values) / (4 * jnp.pi * grid.radius ** 2)


def _align_exact(prediction: xr.Dataset, target: xr.Dataset) -> None:
    for name in ("time", "level", "longitude", "latitude"):
        if name not in prediction.coords or name not in target.coords:
            raise ValueError(f"prediction and target must both have {name} coordinates")
        a, b = prediction[name].values, target[name].values
        # Never let xarray's implicit inner join discard grid cells or valid times.
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"prediction/target {name} coordinates must match exactly")


def _array(dataset: xr.Dataset, variable: str) -> np.ndarray:
    if variable not in dataset:
        raise ValueError(f"missing evaluation variable {variable}")
    field = dataset[variable]
    if set(field.dims) != {"time", "level", "longitude", "latitude"}:
        raise ValueError(f"{variable} requires time, level, longitude, latitude dimensions")
    array = field.transpose("time", "level", "longitude", "latitude").values
    if not np.isfinite(array).all():
        raise ValueError(f"{variable} contains non-finite evaluation values")
    return array


def _climatology_array(climatology: xr.Dataset, target: xr.Dataset, variable: str) -> np.ndarray:
    if variable not in climatology:
        raise ValueError(f"climatology is missing {variable}")
    for name in ("level", "longitude", "latitude"):
        if name not in climatology.coords or not np.array_equal(climatology[name], target[name]):
            raise ValueError(f"climatology {name} coordinates must match target exactly")
    field = climatology[variable]
    if "time" in field.dims:
        if not np.array_equal(climatology.time, target.time):
            raise ValueError("time-dependent climatology must match forecast valid times exactly")
        return _array(climatology, variable)
    if set(field.dims) != {"level", "longitude", "latitude"}:
        raise ValueError("climatology must be static or explicitly selected at forecast valid times")
    values = field.transpose("level", "longitude", "latitude").values
    if not np.isfinite(values).all():
        raise ValueError(f"climatology {variable} contains non-finite values")
    return values[np.newaxis]


def score_predictions(
    prediction: xr.Dataset,
    target: xr.Dataset,
    data_coords: Any,
    *,
    initial_time: np.datetime64,
    variables: list[str] | None = None,
    climatology: xr.Dataset | None = None,
) -> xr.Dataset:
    """Return per-variable, per-pressure-level, per-positive-lead RMSE and ACC.

    ACC is the uncentered, quadrature-weighted correlation of anomalies relative
    to a supplied climatology. A zero anomaly norm produces NaN (undefined ACC).
    No climatology is inferred from the verification sample.
    """
    _align_exact(prediction, target)
    # Native coordinates are checked in addition to prediction/target alignment.
    validate_coordinates(target, type("CoordinateOwner", (), {"data_coords": data_coords})())
    times = target.time.values
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("evaluation time must use datetime64 valid times")
    leads = times - np.datetime64(initial_time)
    if np.any(leads <= np.timedelta64(0, "ns")) or np.any(np.diff(leads) <= np.timedelta64(0, "ns")):
        raise ValueError("evaluation requires strictly increasing positive leads; lead zero is excluded")
    variables = list(target.data_vars) if variables is None else variables
    result = xr.Dataset(coords={"lead_time": leads, "level": target.level.values})
    result = result.assign_coords(valid_time=("lead_time", times))
    mse_by_var = {}
    for variable in variables:
        predicted, observed = _array(prediction, variable), _array(target, variable)
        mse = spherical_mean(jnp.square(jnp.asarray(predicted) - jnp.asarray(observed)), data_coords)
        mse_by_var[variable] = mse
        units = target[variable].attrs.get("units", "unspecified (same as input variable)")
        result[f"{variable}_rmse"] = (("lead_time", "level"), np.asarray(jnp.sqrt(mse)))
        result[f"{variable}_rmse"].attrs.update(units=units, reduction="NeuralGCM nodal_surface_mean")
        if climatology is not None:
            climate = _climatology_array(climatology, target, variable)
            pa, ta = jnp.asarray(predicted - climate), jnp.asarray(observed - climate)
            numerator = spherical_mean(pa * ta, data_coords)
            denominator = jnp.sqrt(spherical_mean(pa * pa, data_coords) * spherical_mean(ta * ta, data_coords))
            acc = jnp.where(denominator > 0, numerator / denominator, jnp.nan)
            result[f"{variable}_acc"] = (("lead_time", "level"), np.asarray(acc))
            result[f"{variable}_acc"].attrs.update(units="1", zero_anomaly_norm="undefined (NaN)")
    u, v = "u_component_of_wind", "v_component_of_wind"
    if u in mse_by_var and v in mse_by_var:
        result["wind_vector_rmse"] = (("lead_time", "level"), np.asarray(jnp.sqrt(mse_by_var[u] + mse_by_var[v])))
        result.wind_vector_rmse.attrs["units"] = target[u].attrs.get("units", "m s-1")
    # Same variable/level choices and Q scaling as official weatherbench2_rmse_metrics.
    # We deliberately label these native-grid selections, not the paper benchmark.
    for name, source, level, scale, units in (
        ("T850", "temperature_rmse", 850, 1, "K"),
        ("Z500", "geopotential_rmse", 500, 1, "m2 s-2"),
        ("UV700", "wind_vector_rmse", 700, 1, "m s-1"),
        ("Q700", "specific_humidity_rmse", 700, 1000, "g kg-1"),
    ):
        if source in result and level in result.level.values:
            result[f"native_{name}_rmse"] = result[source].sel(level=level, drop=True) * scale
            result[f"native_{name}_rmse"].attrs.update(units=units, level_hpa=level)
    result.attrs.update(
        weighting="Dinosaur checkpoint horizontal.integrate / (4*pi*radius**2)",
        protocol="Native-grid deterministic single-initialization; positive forecast leads only",
        reference="neuralgcm/legacy/reference_code/metrics.py:RMSE and metrics_util.py:nodal_surface_mean",
        benchmark_scope="Not a full reproduction of the paper's WeatherBench2 benchmark",
    )
    return result


@dataclass
class EvaluationResult:
    predictions: xr.Dataset
    metrics: xr.Dataset
    metadata: dict

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Write decoded forecasts, metric arrays, and portable JSON records."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"predictions": directory / "predictions.nc", "metrics": directory / "metrics.nc",
                 "summary": directory / "metrics.json"}
        self.predictions.to_netcdf(paths["predictions"])
        self.metrics.to_netcdf(paths["metrics"])
        rows = []
        for name, field in self.metrics.data_vars.items():
            for i, lead in enumerate(self.metrics.lead_time.values):
                per_lead = field.isel(lead_time=i)
                levels = per_lead.level.values if "level" in per_lead.dims else [None]
                for k, level in enumerate(levels):
                    value = float(per_lead.values[k] if level is not None else per_lead.values)
                    rows.append({"metric": name, "lead_hours": float(lead / np.timedelta64(1, "h")),
                                 "level_hpa": None if level is None else float(level),
                                 "value": value if np.isfinite(value) else None,
                                 "units": field.attrs.get("units")})
        paths["summary"].write_text(json.dumps({"metadata": self.metadata, "metrics": rows},
                                              indent=2, allow_nan=False) + "\n")
        return paths


def evaluate(
    model: Any,
    dataset: xr.Dataset,
    forecast_steps: int,
    output_interval: Any = "6h",
    climatology: xr.Dataset | None = None,
    *,
    start_index: int = 0,
    forcing_mode: str = "persistence",
) -> EvaluationResult:
    """Run real NeuralGCM encode/unroll/decode and evaluate future valid times."""
    window = make_training_window(model, dataset, forecast_steps, output_interval,
                                  start_index=start_index, forcing_mode=forcing_mode)
    state = model.encode(window.inputs, window.initial_forcings)
    _, outputs = model.unroll(state, window.forcings, steps=forecast_steps,
                              timedelta=window.output_interval, start_with_input=False)
    predictions = model.data_to_xarray(outputs, times=window.valid_times)
    # API coordinates can differ by harmless serialization rounding. Canonicalize
    # only after checking the complete axes against the checkpoint; no regridding.
    validate_coordinates(predictions, model)
    predictions = predictions.assign_coords({name: window.target_dataset[name]
                                              for name in ("level", "longitude", "latitude")})
    metrics = score_predictions(predictions, window.target_dataset, model.data_coords,
                                initial_time=window.initial_time, variables=model.input_variables,
                                climatology=climatology)
    metadata = {
        "initial_time": str(window.initial_time), "forecast_steps": int(forecast_steps),
        "output_interval_hours": float(window.output_interval / np.timedelta64(1, "h")),
        "forcing_mode": forcing_mode,
        "forcing_protocol": "Initial SST/sea ice persisted" if forcing_mode == "persistence" else
                            "Observed future SST/sea ice prescribed; hindcast boundary conditions",
        "weighting": metrics.attrs["weighting"], "benchmark_scope": metrics.attrs["benchmark_scope"],
        "acc_climatology_supplied": climatology is not None,
    }
    return EvaluationResult(predictions, metrics, metadata)


def evaluate_many(
    model: Any,
    dataset: xr.Dataset,
    forecast_steps: int,
    output_interval: Any = "6h",
    climatology: xr.Dataset | None = None,
    *,
    stride: int = 1,
    forcing_mode: str = "persistence",
    store_predictions: bool = True,
) -> EvaluationResult:
    """Evaluate all complete windows and aggregate *MSE before taking sqrt*.

    The initialization stride is measured in dataset time samples. Missing target
    times inside an otherwise complete horizon are errors, not silently omitted
    samples. ACC is averaged over finite per-initialization anomaly correlations;
    each metric includes its effective initialization count. Static climatology
    is reused; a time-indexed climatology is selected at each window's valid times.
    Forecast storage is optional because full ERA5-sized trajectories are large.
    """
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if isinstance(forecast_steps, bool) or not isinstance(forecast_steps, (int, np.integer)) or forecast_steps < 1:
        raise ValueError("forecast_steps must be a positive integer")
    times = _require_time(dataset)
    interval = output_timedelta(model, output_interval)
    starts = [i for i in range(0, len(times), stride) if times[i] + forecast_steps * interval <= times[-1]]
    if not starts:
        raise ValueError("dataset contains no complete forecast window")
    totals, counts, forecasts, initializations = {}, {}, [], []
    template, metadata = None, None
    for start in starts:
        window_climate = climatology
        if climatology is not None and "time" in climatology.dims:
            valid_times = times[start] + np.arange(1, forecast_steps + 1) * interval
            try:
                window_climate = climatology.sel(time=valid_times)
            except KeyError as exc:
                raise ValueError("climatology is missing exact forecast valid times") from exc
        current = evaluate(model, dataset, forecast_steps, interval, window_climate,
                           start_index=start, forcing_mode=forcing_mode)
        if template is None:
            template = current.metrics.drop_vars("valid_time")
            metadata = dict(current.metadata)
        for name, field in current.metrics.data_vars.items():
            values = np.asarray(field.values, dtype=np.float64)
            finite = np.isfinite(values)
            contribution = values ** 2 if name.endswith("_rmse") else values
            totals[name] = totals.get(name, 0) + np.where(finite, contribution, 0)
            counts[name] = counts.get(name, 0) + finite.astype(np.int64)
        initializations.append(times[start])
        if store_predictions:
            forecast = current.predictions.rename(time="lead_time")
            forecast = forecast.assign_coords(lead_time=current.metrics.lead_time.values)
            forecast = forecast.assign_coords(valid_time=("lead_time", current.predictions.time.values))
            forecasts.append(forecast.expand_dims(initialization_time=[times[start]]))
    aggregate = template.copy(deep=True)
    for name in template.data_vars:
        denominator = counts[name]
        values = np.divide(totals[name], denominator, out=np.full_like(totals[name], np.nan), where=denominator > 0)
        if name.endswith("_rmse"):
            values = np.sqrt(values)
        aggregate[name].values = values
        aggregate[f"{name}_initialization_count"] = (aggregate[name].dims, denominator)
        aggregate[f"{name}_initialization_count"].attrs["units"] = "1"
    aggregate.attrs.update(
        protocol="Native-grid deterministic multiple-initialization; positive forecast leads only",
        initialization_count=len(starts), rmse_aggregation="sqrt(mean(initialization MSE))",
        acc_aggregation="mean of finite per-initialization anomaly correlations",
    )
    predictions = (xr.concat(forecasts, dim="initialization_time", coords="different", compat="equals") if store_predictions else
                   xr.Dataset(coords={"initialization_time": np.asarray(initializations)}))
    metadata.pop("initial_time")
    metadata.update(initialization_count=len(starts), initialization_start=str(initializations[0]),
                    initialization_end=str(initializations[-1]), initialization_stride=int(stride),
                    store_predictions=store_predictions, rmse_aggregation="sqrt(mean(initialization MSE))",
                    acc_aggregation="mean of finite per-initialization anomaly correlations")
    return EvaluationResult(predictions, aggregate, metadata)
