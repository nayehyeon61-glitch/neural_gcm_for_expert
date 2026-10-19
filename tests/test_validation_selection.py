"""Hold-out checkpoint selection uses complete pooled validation forecasts."""
from argparse import Namespace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from neuralgcm_expert.evaluation import complete_forecast_starts, normalized_forecast_mse


def aggregate(rmse=1.0, count=2):
    return xr.Dataset(
        {"temperature_rmse": (("lead_time", "level"), np.full((2, 2), rmse)),
         "temperature_rmse_initialization_count": (("lead_time", "level"), np.full((2, 2), count)),
         # These overlapping diagnostics must never get counted a second time.
         "wind_vector_rmse": (("lead_time", "level"), np.full((2, 2), 999.0)),
         "native_T850_rmse": ("lead_time", np.full(2, 999.0))},
        coords={"lead_time": np.arange(1, 3)*np.timedelta64(24, "h"), "level": [500, 850]},
        attrs={"initialization_count": 2},
    )


def test_pooled_normalized_score_matches_rollout_mse_not_mean_rmse():
    # Initialization errors 1 and 3 imply MSE 5, not square(mean(RMSE)) = 4.
    metrics = aggregate(np.sqrt(5))
    assert normalized_forecast_mse(metrics, {"temperature": 2}) == pytest.approx(5/4)
    assert normalized_forecast_mse(metrics, {"temperature": 2}) != pytest.approx(4/4)


def test_normalized_score_averages_variables_leads_and_levels():
    metrics = aggregate()
    metrics["temperature_rmse"].values = np.array([[1., 2.], [3., 4.]])
    metrics["specific_humidity_rmse"] = (("lead_time", "level"), np.full((2, 2), .002))
    metrics["specific_humidity_rmse_initialization_count"] = (("lead_time", "level"), np.full((2, 2), 2))
    expected = (np.mean(np.square([1, 2, 3, 4])) / 4 + 4) / 2
    assert normalized_forecast_mse(metrics, {"temperature": 2, "specific_humidity": .001}) == pytest.approx(expected)


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan, [1, 2], None, "invalid"])
def test_reject_bad_scales(scale):
    with pytest.raises(ValueError, match="finite positive scalar"):
        normalized_forecast_mse(aggregate(), {"temperature": scale})


def test_reject_incomplete_or_nonfinite_validation():
    with pytest.raises(ValueError, match="incomplete initialization"):
        normalized_forecast_mse(aggregate(count=1), {"temperature": 1})
    with pytest.raises(ValueError, match="missing initialization"):
        normalized_forecast_mse(aggregate().drop_vars("temperature_rmse_initialization_count"), {"temperature": 1})
    with pytest.raises(ValueError, match="finite nonnegative"):
        normalized_forecast_mse(aggregate(np.nan), {"temperature": 1})
    with pytest.raises(ValueError, match="Scale names"):
        normalized_forecast_mse(aggregate(), {"specific_humidity": 1})


def time_dataset(start="2000-01-01", length=6):
    return xr.Dataset(coords={"time": np.datetime64(start)+np.arange(length)*np.timedelta64(1, "D")})


def test_complete_window_preflight_checks_stride_horizon_and_gaps():
    model = SimpleNamespace(timestep=np.timedelta64(1, "h"))
    ds = time_dataset()
    assert complete_forecast_starts(model, ds, 2, "24h") == [0, 1, 2, 3]
    assert complete_forecast_starts(model, ds, 2, "24h", stride=2) == [0, 2]
    with pytest.raises(ValueError, match="no complete forecast"):
        complete_forecast_starts(model, ds, 6, "24h")
    with pytest.raises(ValueError, match="missing exact forecast"):
        complete_forecast_starts(model, ds.isel(time=[0, 2, 3, 4, 5]), 2, "24h")
    with pytest.raises(ValueError, match="positive integer"):
        complete_forecast_starts(model, ds, 2, "24h", stride=0)


def train_args(tmp_path, **changes):
    values = dict(checkpoint="initial.pkl", train_data="train", validation_data="valid",
                  output=str(tmp_path), epochs=3, steps=1, validation_steps=2,
                  interval="24h", stride=1, validation_stride=2, learning_rate=1e-4,
                  scope="physics", scales=None, native_scales=None, tendency_weight=0.,
                  acceleration_weight=0., residual_weight=1e-6, forcing_mode="persistence", seed=0)
    values.update(changes)
    return Namespace(**values)


def test_train_selects_all_start_score_and_keeps_optimizer_state(monkeypatch, tmp_path):
    from neuralgcm_expert import cli, data, evaluation, model as model_module, training

    train_ds, valid_ds = time_dataset(length=4), time_dataset("2001-01-01")
    fake_model = SimpleNamespace(params={"physics": {"w": np.array(0.)}}, expert_metadata={},
                                 timestep=np.timedelta64(1, "h"), input_variables=["temperature"],
                                 data_coords=SimpleNamespace(horizontal=None))
    monkeypatch.setattr(model_module, "load_checkpoint", lambda path: {})
    monkeypatch.setattr(model_module, "restore_checkpoint", lambda checkpoint: fake_model)
    monkeypatch.setattr(data, "open_dataset", lambda path: train_ds if path == "train" else valid_ds)
    monkeypatch.setattr(evaluation, "quadrature_weights", lambda grid: np.array(1.))
    monkeypatch.setattr(training, "trainable_mask", lambda params, scope: {"physics": {"w": True}})
    monkeypatch.setattr(data, "make_training_window", lambda *a, **k: {})
    monkeypatch.setattr(training, "window_batch", lambda window: {})
    monkeypatch.setattr(training, "with_params", lambda model, params: SimpleNamespace(**{**vars(model), "params": params}))
    scale_calls = []

    def scales(ds, variables):
        scale_calls.append(ds)
        return {"temperature": 2.0}

    monkeypatch.setattr(training, "estimate_scales", scales)
    optimizer_calls, states = [], []

    def make_step(*args, **kwargs):
        optimizer_calls.append(kwargs)

        def update(params, state, batch):
            states.append(state)
            return {"physics": {"w": params["physics"]["w"]+1}}, state+1, {"accepted": 1., "total": 1.}

        return None, 0, update

    monkeypatch.setattr(training, "make_train_step", make_step)
    evaluations = []
    scores = iter([5., 1., 3.])

    def evaluate(current, ds, **kwargs):
        assert ds is valid_ds
        evaluations.append(kwargs)
        score = next(scores)
        return SimpleNamespace(metrics=aggregate(np.sqrt(score)*2),
                               metadata={"initialization_start": "2001-01-01", "initialization_end": "2001-01-03"},
                               save=lambda path: None)

    monkeypatch.setattr(evaluation, "evaluate_many", evaluate)
    saved = []
    monkeypatch.setattr(model_module, "save_checkpoint", lambda model, path, checkpoint: saved.append((path.name, float(model.params["physics"]["w"]))))
    cli.train(train_args(tmp_path))
    assert len(scale_calls) == 1 and scale_calls[0] is train_ds
    assert len(optimizer_calls) == 1
    assert states == list(range(9))  # Adam state must survive epoch boundaries.
    assert all(call["forecast_steps"] == 2 and call["stride"] == 2
               and call["store_predictions"] is False for call in evaluations)
    assert saved == [("last.pkl", 3), ("best.pkl", 3), ("last.pkl", 6),
                     ("best.pkl", 6), ("last.pkl", 9)]
    selected = json.loads((tmp_path/"validation-selection.json").read_text())
    assert selected["best_epoch"] == 2
    assert selected["best_normalized_mse"] == pytest.approx(1.)
    assert selected["current_normalized_mse"] == pytest.approx(3.)
    assert selected["initialization_count"] == 2
    assert selected["forecast_horizon_hours"] == 48
    assert selected["test_data_used"] is False


def test_empty_validation_windows_fail_before_optimizer_setup(monkeypatch, tmp_path):
    from neuralgcm_expert import cli, data, model as model_module, training

    fake_model = SimpleNamespace(timestep=np.timedelta64(1, "h"))
    monkeypatch.setattr(model_module, "load_checkpoint", lambda path: {})
    monkeypatch.setattr(model_module, "restore_checkpoint", lambda checkpoint: fake_model)
    monkeypatch.setattr(data, "open_dataset", lambda path: time_dataset() if path == "train" else time_dataset("2001-01-01", 2))
    monkeypatch.setattr(training, "make_train_step", lambda *a, **k: pytest.fail("optimizer setup must follow preflight"))
    with pytest.raises(ValueError, match="no complete forecast"):
        cli.train(train_args(tmp_path))
