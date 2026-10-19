"""Pairing and arithmetic checks for the standalone evaluation comparison CLI."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


SPEC = importlib.util.spec_from_file_location(
    "compare_metrics", Path(__file__).resolve().parents[1] / "scripts" / "compare_metrics.py"
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def save_fixture(directory, value=2.0, *, times=None, mutate=None):
    directory.mkdir()
    times = (np.asarray(["2020-01-01", "2020-01-02", "2020-01-04"], dtype="datetime64[ns]")
             if times is None else times)
    metadata = dict(forecast_steps=5, output_interval_hours=24.0, forcing_mode="persistence",
                    weighting="official quadrature", initialization_count=len(times),
                    initialization_stride=1, initialization_start=str(times[0]),
                    initialization_end=str(times[-1]), rmse_aggregation="sqrt(mean(initialization MSE))")
    rows = []
    for day in range(1, 6):
        for name, level in (("temperature_rmse", 850.0), ("native_T850_rmse", None)):
            rows.append(dict(metric=name, lead_hours=24.0 * day, level_hpa=level, value=value, units="K"))
            rows.append(dict(metric=name + "_initialization_count", lead_hours=24.0 * day,
                             level_hpa=level, value=float(len(times)), units="1"))
    payload = {"metadata": metadata, "metrics": rows}
    if mutate is not None:
        mutate(payload)
    (directory / "metrics.json").write_text(json.dumps(payload))
    xr.Dataset(coords={"initialization_time": times}).to_netcdf(directory / "predictions.nc")
    return directory


def test_paired_improvement_and_csv_summary(tmp_path, capsys):
    baseline = save_fixture(tmp_path / "baseline", 2.0)
    candidate = save_fixture(tmp_path / "candidate", 1.5)
    output = tmp_path / "report" / "comparison.csv"
    assert comparison.main(["--baseline", str(baseline), "--candidate", str(candidate),
                            "--output", str(output)]) == 0
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {float(row["improvement_percent"]) for row in rows} == {25.0}
    assert {row["initialization_count"] for row in rows} == {"3"}
    summary = capsys.readouterr().out
    for day in (1, 3, 5):
        assert f"T850 day {day}: 2 -> 1.5 K (+25.000%)" in summary
    assert "statistical significance is not assessed" in summary


@pytest.mark.parametrize("baseline_value,candidate_value,expected", [(0, 1, None), (2, 3, -50.0)])
def test_zero_baseline_and_degradation(tmp_path, baseline_value, candidate_value, expected):
    baseline = save_fixture(tmp_path / "baseline", baseline_value)
    candidate = save_fixture(tmp_path / "candidate", candidate_value)
    assert all(row["improvement_percent"] == expected
               for row in comparison.compare_directories(baseline, candidate))


def test_rejects_different_middle_initialization_with_same_metadata(tmp_path):
    baseline = save_fixture(tmp_path / "baseline")
    candidate = save_fixture(tmp_path / "candidate", times=np.asarray(
        ["2020-01-01", "2020-01-03", "2020-01-04"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="initialization_time arrays must match exactly"):
        comparison.compare_directories(baseline, candidate)


@pytest.mark.parametrize("field,value", [("forcing_mode", "dynamic"), ("weighting", "cosine"),
                                          ("initialization_stride", 2)])
def test_rejects_protocol_mismatch(tmp_path, field, value):
    baseline = save_fixture(tmp_path / "baseline")
    candidate = save_fixture(tmp_path / "candidate", mutate=lambda payload: payload["metadata"].update({field: value}))
    with pytest.raises(ValueError, match=f"protocol mismatch for {field}"):
        comparison.compare_directories(baseline, candidate)


@pytest.mark.parametrize("mutation,reason", [
    (lambda data: data["metrics"][0].update(value=None), "RMSE must be finite"),
    (lambda data: data["metrics"][1].update(value=2), "initialization count is missing or incomplete"),
    (lambda data: data["metrics"][0].update(units="degC"), "units must match exactly"),
    (lambda data: data["metrics"].pop(0), "missing or unexpected lead/level rows"),
    (lambda data: data["metrics"].append(data["metrics"][0]), "duplicate metric row"),
])
def test_rejects_incomplete_or_incompatible_metrics(tmp_path, mutation, reason):
    baseline = save_fixture(tmp_path / "baseline")
    candidate = save_fixture(tmp_path / "candidate", mutate=mutation)
    with pytest.raises(ValueError, match=reason):
        comparison.compare_directories(baseline, candidate)
