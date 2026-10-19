#!/usr/bin/env python3
"""Compare paired native-grid RMSE from two ``evaluate --all-starts`` directories.

This reports descriptive differences, not statistical significance. Prediction
fields need not be stored: the initialization_time coordinate is sufficient.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr


PROTOCOL_FIELDS = (
    "forecast_steps", "output_interval_hours", "forcing_mode", "weighting",
    "initialization_count", "initialization_stride", "initialization_start",
    "initialization_end", "rmse_aggregation",
)
CSV_FIELDS = (
    "metric", "lead_hours", "level_hpa", "units", "baseline_rmse",
    "candidate_rmse", "improvement_percent", "initialization_count",
)


def _positive_number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a positive number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{description} must be a positive finite number")
    return value


def _load(directory: str | Path):
    directory = Path(directory)
    payload = json.loads((directory / "metrics.json").read_text())
    metadata = payload["metadata"]
    for name in PROTOCOL_FIELDS:
        if name not in metadata or metadata[name] is None:
            raise ValueError(f"{directory}: missing protocol metadata {name}")
    for name in ("forecast_steps", "initialization_count", "initialization_stride"):
        value = _positive_number(metadata[name], name)
        if not isinstance(value, int):
            raise ValueError(f"{name} must be a positive integer")
    interval = _positive_number(metadata["output_interval_hours"], "output_interval_hours")
    if metadata["rmse_aggregation"] != "sqrt(mean(initialization MSE))":
        raise ValueError("RMSE aggregation must be sqrt(mean(initialization MSE))")

    with xr.open_dataset(directory / "predictions.nc") as dataset:
        if "initialization_time" not in dataset.coords:
            raise ValueError(f"{directory}: missing initialization_time coordinate; use evaluate --all-starts")
        coordinate = dataset["initialization_time"]
        if coordinate.dims != ("initialization_time",):
            raise ValueError("initialization_time must be a one-dimensional coordinate")
        initializations = coordinate.values.copy()
    if not np.issubdtype(initializations.dtype, np.datetime64):
        raise ValueError("initialization_time must contain datetime64 values")
    if len(initializations) != metadata["initialization_count"]:
        raise ValueError("initialization count does not match predictions.nc")
    if np.isnat(initializations).any() or np.any(np.diff(initializations) <= np.timedelta64(0, "ns")):
        raise ValueError("initialization_time must be finite, unique and strictly increasing")
    for name, value in (("initialization_start", initializations[0]),
                        ("initialization_end", initializations[-1])):
        if np.datetime64(metadata[name]) != value:
            raise ValueError(f"{name} does not match predictions.nc")

    records = {}
    for row in payload["metrics"]:
        name = row["metric"]
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be nonempty strings")
        lead = _positive_number(row["lead_hours"], f"{name} lead_hours")
        level = row["level_hpa"]
        if level is not None:
            _positive_number(level, f"{name} level_hpa")
        key = (name, lead, level)
        if key in records:
            raise ValueError(f"duplicate metric row: {key}")
        records[key] = row
    rmse_keys = [key for key in records if key[0].endswith("_rmse")]
    if not rmse_keys:
        raise ValueError("no RMSE metrics found")

    expected_leads = set(interval * step for step in range(1, metadata["forecast_steps"] + 1))
    for name in {key[0] for key in rmse_keys}:
        keys = [key for key in rmse_keys if key[0] == name]
        levels = {key[2] for key in keys}
        if len(levels) > 1 and None in levels:
            raise ValueError(f"{name}: mixed scalar and pressure-level metrics")
        if {(key[1], key[2]) for key in keys} != {(lead, level) for lead in expected_leads for level in levels}:
            raise ValueError(f"{name}: missing or unexpected lead/level rows")
        for key in keys:
            row = records[key]
            value = row["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{key}: RMSE must be finite and nonnegative; failed forecasts cannot be skipped")
            if not isinstance(row.get("units"), str) or not row["units"]:
                raise ValueError(f"{key}: missing RMSE units")
            count_key = (name + "_initialization_count", key[1], key[2])
            count = records.get(count_key, {}).get("value")
            if isinstance(count, bool) or count != metadata["initialization_count"]:
                raise ValueError(f"{key}: initialization count is missing or incomplete")
    return metadata, initializations, records


def compare_directories(baseline: str | Path, candidate: str | Path) -> list[dict]:
    """Require identical protocol and samples before comparing any RMSE row."""
    baseline_meta, baseline_times, baseline_rows = _load(baseline)
    candidate_meta, candidate_times, candidate_rows = _load(candidate)
    for name in PROTOCOL_FIELDS:
        if baseline_meta[name] != candidate_meta[name]:
            raise ValueError(f"protocol mismatch for {name}: {baseline_meta[name]!r} != {candidate_meta[name]!r}")
    if not np.array_equal(baseline_times, candidate_times):
        raise ValueError("initialization_time arrays must match exactly for paired comparison")
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("metric names, leads and pressure levels must match exactly")
    for key in baseline_rows:
        if baseline_rows[key].get("units") != candidate_rows[key].get("units"):
            raise ValueError(f"metric units must match exactly: {key}")

    result = []
    keys = sorted(baseline_rows, key=lambda key: (key[0], key[1], -1 if key[2] is None else key[2]))
    for key in keys:
        if not key[0].endswith("_rmse"):
            continue
        baseline_value = baseline_rows[key]["value"]
        candidate_value = candidate_rows[key]["value"]
        result.append({
            "metric": key[0], "lead_hours": key[1], "level_hpa": key[2],
            "units": baseline_rows[key]["units"], "baseline_rmse": baseline_value,
            "candidate_rmse": candidate_value,
            "improvement_percent": (100 * (baseline_value - candidate_value) / baseline_value
                                    if baseline_value > 0 else None),
            "initialization_count": baseline_meta["initialization_count"],
        })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline evaluation directory")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate evaluation directory")
    parser.add_argument("--output", type=Path, required=True, help="Comparison CSV file")
    args = parser.parse_args(argv)
    try:
        rows = compare_directories(args.baseline, args.candidate)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} paired RMSE comparisons to {args.output}")
    print("Native-grid RMSE; positive improvement means lower candidate RMSE.")
    summary_rows = [row for row in rows if row["metric"] in {
        "native_T850_rmse", "native_Z500_rmse", "native_UV700_rmse", "native_Q700_rmse"
    } and row["lead_hours"] in (24, 72, 120)]
    for row in summary_rows:
        improvement = (f"{row['improvement_percent']:+.3f}%" if row["improvement_percent"] is not None
                       else "undefined (zero baseline)")
        name = row["metric"].removeprefix("native_").removesuffix("_rmse")
        print(f"  {name} day {row['lead_hours'] / 24:g}: "
              f"{row['baseline_rmse']:.6g} -> {row['candidate_rmse']:.6g} "
              f"{row['units']} ({improvement})")
    if not summary_rows:
        print("  No native T850/Z500/UV700/Q700 scores at 1, 3 or 5 days in this evaluation.")
    print("These are descriptive paired differences; statistical significance is not assessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
