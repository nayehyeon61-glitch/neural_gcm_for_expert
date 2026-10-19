"""Mocked orchestration checks; no training, rollout or GPU allocation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "run_comparison", Path(__file__).resolve().parents[1] / "scripts" / "run_comparison.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def mocked_runtime(monkeypatch):
    calls, inspected = [], []
    monkeypatch.setattr(runner, "check_backend", lambda allow_cpu: None)
    monkeypatch.setattr(runner, "restore_checkpoint", lambda path: object())
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "test-commit\n")
    monkeypatch.setattr(runner, "run_cli", lambda *args: calls.append(list(map(str, args))))

    def inspect(path, model, steps, interval):
        inspected.append((str(path), steps, interval))
        year = {"train.nc": 2018, "valid.nc": 2019, "test.nc": 2020}[Path(path).name]
        return {"path": str(path), "start": f"{year}-01-01", "end": f"{year}-12-31",
                "sample_interval_hours": 24, "samples": 365,
                "input_variables": ["temperature"], "forcing_variables": ["sea_surface_temperature"]}

    monkeypatch.setattr(runner, "inspect_data", inspect)
    return calls, inspected


def train_arguments(tmp_path, *, output=None):
    (tmp_path / "official.pkl").write_bytes(b"mock official checkpoint")
    return ["train", "--checkpoint", str(tmp_path / "official.pkl"),
            "--train-data", str(tmp_path / "train.nc"),
            "--validation-data", str(tmp_path / "valid.nc"),
            "--output", str(output or tmp_path / "run"), "--include-memoryless", "--allow-cpu"]


def flag(command, name):
    return command[command.index(name) + 1]


def test_training_uses_common_scales_and_no_test_data(tmp_path, mocked_runtime):
    calls, inspected = mocked_runtime
    runner.main(train_arguments(tmp_path))
    assert [Path(item[0]).name for item in inspected] == ["train.nc", "valid.nc"]
    assert [command[0] for command in calls] == ["init"] * 3 + ["train"] * 3
    training = calls[3:]
    assert "--scales" not in training[0]
    common = str(tmp_path / "run" / "baseline_finetuned" / "scales.json")
    assert flag(training[1], "--scales") == common
    assert flag(training[2], "--scales") == common
    for command in training:
        assert flag(command, "--steps") == "1"
        assert flag(command, "--validation-steps") == "5"
        assert flag(command, "--validation-stride") == "7"
        assert flag(command, "--forcing-mode") == "persistence"
        assert flag(command, "--scope") == "physics"
        assert flag(command, "--tendency-weight") == "0"
        assert flag(command, "--acceleration-weight") == "0"
    protocol = json.loads((tmp_path / "run" / "protocol.json").read_text())
    assert protocol["phase"] == "trained"
    assert protocol["modes"] == ["baseline", "memoryless", "experts"]
    assert "test" not in protocol


def test_training_refuses_nonempty_output(tmp_path, mocked_runtime):
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing-results").write_text("preserve")
    with pytest.raises(FileExistsError, match="not empty"):
        runner.main(train_arguments(tmp_path))
    assert mocked_runtime == ([], [])
    assert (output / "existing-results").read_text() == "preserve"


def prepare_test_run(tmp_path):
    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    protocol = {
        "phase": "trained", "train": {"end": "2018-12-31", "sample_interval_hours": 24},
        "validation": {"start": "2019-01-01", "end": "2019-12-31"},
        "modes": ["baseline", "experts"], "evaluation_steps": 5,
        "interval": "24h", "test_stride": 1, "forcing_mode": "persistence",
    }
    (output / "protocol.json").write_text(json.dumps(protocol))
    for relative in ("checkpoints/baseline-init.pkl", "baseline_finetuned/best.pkl", "experts/best.pkl"):
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"mock checkpoint")
    return output, protocol


def test_test_phase_uses_frozen_choices_and_paired_comparisons(tmp_path, mocked_runtime, monkeypatch):
    calls, inspected = mocked_runtime
    output, _ = prepare_test_run(tmp_path)
    comparisons = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: comparisons.append(command))
    runner.main(["test", "--run", str(output), "--test-data", str(tmp_path / "test.nc"), "--allow-cpu"])
    assert [Path(item[0]).name for item in inspected] == ["test.nc"]
    assert len(calls) == 3
    for command in calls:
        assert command[0] == "evaluate"
        assert "--all-starts" in command
        assert "--store-predictions" not in command
        assert flag(command, "--steps") == "5"
        assert flag(command, "--stride") == "1"
        assert flag(command, "--forcing-mode") == "persistence"
    assert len(comparisons) == 2
    assert {Path(flag(command, "--baseline")).name for command in comparisons} == {
        "baseline_pretrained", "baseline_finetuned"}
    assert all(flag(command, "--candidate") == str(output / "test" / "experts") for command in comparisons)
    assert json.loads((output / "protocol.json").read_text())["phase"] == "tested"


@pytest.mark.parametrize("case,error", [("unfinished", "completed training"),
                                       ("overlap", "chronological"), ("existing", "already exists")])
def test_test_phase_guards(tmp_path, mocked_runtime, case, error):
    output, protocol = prepare_test_run(tmp_path)
    if case == "unfinished":
        protocol["phase"] = "training"
    elif case == "overlap":
        protocol["validation"]["end"] = "2020-01-01"
    elif case == "existing":
        (output / "test").mkdir()
    (output / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises((ValueError, FileExistsError), match=error):
        runner.main(["test", "--run", str(output), "--test-data", str(tmp_path / "test.nc"), "--allow-cpu"])
    assert mocked_runtime[0] == []
    if case != "existing":
        assert not (output / "test").exists()
