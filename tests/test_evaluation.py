"""Evaluation protocol checks on real Dinosaur Gaussian grid quadrature."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from dinosaur import coordinate_systems, spherical_harmonic, typing, vertical_interpolation
import numpy as np
import pytest
import xarray as xr

from neuralgcm_expert.data import make_training_window, open_dataset
from neuralgcm_expert.evaluation import evaluate, evaluate_many, quadrature_weights, score_predictions, spherical_mean


class APIModel:
    """Protocol test double; numerical metrics still use the actual Dinosaur grid."""
    input_variables = ["temperature", "geopotential", "u_component_of_wind",
                       "v_component_of_wind", "specific_humidity"]
    forcing_variables = ["sea_surface_temperature", "sea_ice_cover"]
    timestep = np.timedelta64(1, "h")

    def __init__(self):
        self.data_coords = coordinate_systems.CoordinateSystem(
            spherical_harmonic.Grid.construct(max_wavenumber=3, gaussian_nodes=4, radius=1.0),
            vertical_interpolation.PressureCoordinates([500, 700, 850]),
        )
        self.last_unroll = None

    def _extract(self, dataset, variables, surface=False):
        result = {}
        for name in variables:
            dims = (["time"] if "time" in dataset[name].dims else [])
            dims += [] if surface else ["level"]
            dims += ["longitude", "latitude"]
            result[name] = dataset[name].transpose(*dims).values
            if surface:
                result[name] = np.expand_dims(result[name], axis=-3)
        result["sim_time"] = (dataset.time.values - np.datetime64("2020-01-01")) / np.timedelta64(1, "h")
        return result

    def inputs_from_xarray(self, dataset):
        return self._extract(dataset, self.input_variables)

    def forcings_from_xarray(self, dataset):
        return self._extract(dataset, self.forcing_variables, surface=True)

    def encode(self, inputs, forcings):
        assert np.ndim(inputs["sim_time"]) == 0
        return inputs

    def unroll(self, state, forcings, *, steps, timedelta, start_with_input):
        self.last_unroll = dict(steps=steps, timedelta=timedelta, start_with_input=start_with_input)
        assert start_with_input is False
        offset = np.arange(1, steps + 1) * (timedelta / np.timedelta64(6, "h"))
        outputs = {name: state[name][None] + offset[:, None, None, None] + .25
                   for name in self.input_variables}
        return state, outputs

    def data_to_xarray(self, data, times):
        return xr.Dataset({k: (("time", "level", "longitude", "latitude"), v) for k, v in data.items()},
                          coords=dict(time=times, level=self.data_coords.vertical.centers,
                                      longitude=np.rad2deg(self.data_coords.horizontal.longitudes),
                                      latitude=np.rad2deg(self.data_coords.horizontal.latitudes)))


@pytest.fixture
def model():
    return APIModel()


@pytest.fixture
def dataset(model):
    time = np.datetime64("2020-01-01T00:00:00") + np.arange(4) * np.timedelta64(6, "h")
    shape = (4, 3, *model.data_coords.horizontal.nodal_shape)
    variables = {name: (("time", "level", "longitude", "latitude"),
                        np.broadcast_to(np.arange(4)[:, None, None, None] + i, shape).copy().astype(float))
                 for i, name in enumerate(model.input_variables)}
    for name in model.forcing_variables:
        variables[name] = (("time", "longitude", "latitude"),
                           np.zeros((4, *model.data_coords.horizontal.nodal_shape)))
    return xr.Dataset(variables, coords=dict(time=time, level=model.data_coords.vertical.centers,
                                            longitude=np.rad2deg(model.data_coords.horizontal.longitudes),
                                            latitude=np.rad2deg(model.data_coords.horizontal.latitudes)))


def test_constant_errors_and_official_weatherbench_selections(model, dataset):
    target = dataset[model.input_variables].isel(time=slice(1, 3))
    prediction = target.copy(deep=True)
    offsets = dict(temperature=2, geopotential=9, u_component_of_wind=3,
                   v_component_of_wind=4, specific_humidity=.001)
    for name, offset in offsets.items():
        prediction[name] += offset
    metrics = score_predictions(prediction, target, model.data_coords, initial_time=dataset.time.values[0])
    np.testing.assert_allclose(metrics.temperature_rmse, 2, rtol=1e-6)
    np.testing.assert_allclose(metrics.wind_vector_rmse, 5, rtol=1e-6)
    np.testing.assert_allclose(metrics.native_T850_rmse, 2, rtol=1e-6)
    np.testing.assert_allclose(metrics.native_Z500_rmse, 9, rtol=1e-6)
    np.testing.assert_allclose(metrics.native_UV700_rmse, 5, rtol=1e-6)
    np.testing.assert_allclose(metrics.native_Q700_rmse, 1, rtol=2e-4)
    np.testing.assert_array_equal(metrics.lead_time, [np.timedelta64(6, "h"), np.timedelta64(12, "h")])


def test_gaussian_weights_are_official_not_cosine(model):
    grid = model.data_coords.horizontal
    values = np.broadcast_to(np.arange(grid.latitude_nodes, dtype=float) ** 4, grid.nodal_shape)
    weights = quadrature_weights(grid)
    np.testing.assert_allclose(weights.sum(), 1, rtol=1e-14)
    expected = np.sum(values * weights)
    np.testing.assert_allclose(spherical_mean(values, model.data_coords), expected, rtol=1e-6)
    cosine = np.cos(grid.latitudes)
    cosine_mean = np.sum(values.mean(axis=0) * cosine) / cosine.sum()
    assert abs(cosine_mean - expected) > .01


def test_acc_climatology_and_undefined_zero_norm(model, dataset):
    target = dataset[["temperature"]].isel(time=slice(1, 3))
    climate = target.isel(time=0, drop=True) * 0
    matching = score_predictions(target, target, model.data_coords, initial_time=dataset.time.values[0],
                                  climatology=climate)
    np.testing.assert_allclose(matching.temperature_acc, 1, rtol=1e-6)
    reversed_ = score_predictions(-target, target, model.data_coords, initial_time=dataset.time.values[0],
                                  climatology=climate)
    np.testing.assert_allclose(reversed_.temperature_acc, -1, rtol=1e-6)
    zero = score_predictions(target * 0, target * 0, model.data_coords, initial_time=dataset.time.values[0],
                             climatology=climate)
    assert np.isnan(zero.temperature_acc).all()


def test_window_target_and_forcing_time_contract(model, dataset):
    window = make_training_window(model, dataset, 2, "6h")
    np.testing.assert_array_equal(window.valid_times, dataset.time.values[1:3])
    assert window.inputs["temperature"].shape == (3, *model.data_coords.horizontal.nodal_shape)
    assert window.targets["temperature"].shape[0] == 2
    assert window.forcings["sea_ice_cover"].shape[0] == 1
    dynamic = make_training_window(model, dataset, 2, "6h", forcing_mode="dynamic")
    assert dynamic.forcings["sea_ice_cover"].shape[0] == 3


def test_real_api_call_contract_and_result_persistence(model, dataset, tmp_path):
    result = evaluate(model, dataset, 2, "6h")
    assert model.last_unroll["start_with_input"] is False
    np.testing.assert_allclose(result.metrics.temperature_rmse, .25, rtol=1e-6)
    paths = result.save(tmp_path)
    with open_dataset(paths["predictions"]) as saved:
        np.testing.assert_array_equal(saved.time.values, dataset.time.values[1:3])
    summary = json.loads(paths["summary"].read_text())
    assert summary["metadata"]["forcing_mode"] == "persistence"
    assert all(row["lead_hours"] > 0 for row in summary["metrics"])


def test_multiple_initializations_aggregate_mse_not_rmse(model, dataset, monkeypatch):
    ordinary = model.unroll

    def varying_error(state, *args, **kwargs):
        final, output = ordinary(state, *args, **kwargs)
        # First initialization error 1, second initialization error 3.
        extra = .75 + float(state["sim_time"]) / 3
        return final, {name: value + extra for name, value in output.items()}

    monkeypatch.setattr(model, "unroll", varying_error)
    result = evaluate_many(model, dataset, 2, "6h")
    np.testing.assert_allclose(result.metrics.temperature_rmse, np.sqrt(5), rtol=1e-6)
    np.testing.assert_array_equal(result.metrics.temperature_rmse_initialization_count, 2)
    assert result.predictions.sizes["initialization_time"] == 2
    assert result.predictions.sizes["lead_time"] == 2
    assert set(result.predictions.valid_time.dims) == {"initialization_time", "lead_time"}
    assert result.metadata["initialization_count"] == 2
    compact = evaluate_many(model, dataset, 2, "6h", store_predictions=False)
    assert not compact.predictions.data_vars


@pytest.mark.parametrize("bad_interval", ["0h", "-1h", "90min"])
def test_reject_invalid_output_interval(model, dataset, bad_interval):
    with pytest.raises(ValueError, match="output_interval"):
        make_training_window(model, dataset, 1, bad_interval)


def test_reject_missing_duplicate_and_reordered_times(model, dataset):
    with pytest.raises(ValueError, match="missing exact"):
        make_training_window(model, dataset.isel(time=[0, 2, 3]), 2, "6h")
    for indices in ([0, 1, 1, 3], [0, 2, 1, 3]):
        with pytest.raises(ValueError, match="unique and strictly increasing"):
            make_training_window(model, dataset.isel(time=indices), 2, "6h")


def test_reject_grid_mismatch_missing_values_and_lead_zero(model, dataset):
    with pytest.raises(ValueError, match="latitude coordinate mismatch"):
        make_training_window(model, dataset.isel(latitude=slice(None, None, -1)), 2)
    nonfinite = dataset.copy(deep=True)
    nonfinite["sea_ice_cover"].values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        make_training_window(model, nonfinite, 2)
    target = dataset[["temperature"]].isel(time=slice(0, 2))
    with pytest.raises(ValueError, match="positive leads"):
        score_predictions(target, target, model.data_coords, initial_time=dataset.time.values[0])
    shifted = target.assign_coords(time=target.time.values + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="time coordinates must match exactly"):
        score_predictions(shifted, target, model.data_coords, initial_time=dataset.time.values[0])


def test_reject_climatology_misalignment(model, dataset):
    target = dataset[["temperature"]].isel(time=slice(1, 3))
    climate = target.isel(time=0, drop=True).isel(latitude=slice(None, None, -1))
    with pytest.raises(ValueError, match="climatology latitude"):
        score_predictions(target, target, model.data_coords, initial_time=dataset.time.values[0],
                          climatology=climate)


def test_optional_direct_upstream_rmse_parity(model, dataset, monkeypatch):
    """Run against the unmodified reference when a source checkout is available.

    PyPI omits reference_code. Set NEURALGCM_REFERENCE_DIR to a source checkout's
    neuralgcm/legacy/reference_code directory to enable this independent check.
    """
    source = os.environ.get("NEURALGCM_REFERENCE_DIR")
    if not source or not (Path(source) / "metrics.py").is_file():
        pytest.skip("optional official source reference unavailable in NeuralGCM wheel")
    monkeypatch.syspath_prepend(source)
    upstream_metrics = importlib.import_module("metrics")
    upstream_util = importlib.import_module("metrics_util")
    target = dataset[["temperature"]].isel(time=slice(1, 3))
    prediction = target.copy(deep=True)
    grid = model.data_coords.horizontal
    error = np.broadcast_to(np.arange(grid.latitude_nodes, dtype=float) ** 2, grid.nodal_shape)
    prediction["temperature"] += error
    ours = score_predictions(prediction, target, model.data_coords, initial_time=dataset.time.values[0])
    spec = upstream_util.TrajectorySpec(2, 2, 6, model.data_coords, model.data_coords)
    metric = upstream_metrics.RMSE(spec, time_step=1, level=850, getter=lambda x: x["temperature"])
    pred_rep = typing.TrajectoryRepresentations({"temperature": prediction.temperature.values}, None, None, None)
    target_rep = typing.TrajectoryRepresentations({"temperature": target.temperature.values}, None, None, None)
    expected = metric.evaluate(pred_rep, target_rep)
    np.testing.assert_allclose(ours.temperature_rmse.isel(lead_time=1).sel(level=850), expected, rtol=1e-6)
