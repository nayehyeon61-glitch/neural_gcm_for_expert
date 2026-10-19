"""Opt-in integration test of actual official solver differentiation.

Run RUN_NEURALGCM_INTEGRATION=1 python -m pytest -q tests/test_training.py.
Also set RUN_NEURALGCM_NATIVE_LOSSES=1 to exercise both native derivative losses.
The test is opt-in because compiling the full spherical-harmonic solver and its
gradient is substantially more expensive than the component contract checks.
"""
import os

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_NEURALGCM_INTEGRATION") != "1",
    reason="Set RUN_NEURALGCM_INTEGRATION=1 to compile the real NeuralGCM rollout gradient",
)
def test_official_rollout_gradient_update_and_checkpoint_roundtrip(tmp_path):
    from examples.smoke_test import run_smoke

    native_losses = os.environ.get("RUN_NEURALGCM_NATIVE_LOSSES") == "1"
    report = run_smoke(tmp_path / "trained-smoke.pkl", native_losses=native_losses)
    assert report["metrics"]["accepted"]
    assert report["expert_parameter_change_l1"] > 0
    assert report["frozen_parameter_change_l1"] == 0
    assert report["checkpoint_roundtrip"]
    if native_losses:
        assert report["metrics"]["tendency"] > 0
        assert report["metrics"]["acceleration"] > 0
