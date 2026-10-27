"""Monitoring and the existing trust formula; no CNN training or datasets."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import fakeredis
import pytest
import torch
from shared import state_manager
from h_node import twin_trust, orchestrator, heartbeat_handler


@pytest.fixture
def redis_state(monkeypatch):
    redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(state_manager, "r", redis)
    return redis


def test_heartbeat_updates_status_round_epoch_and_expiry(redis_state):
    state_manager.update_node_status("0", "1", "training", round_id=2, epoch=0)
    key = "fl:hospital:0:node:1"
    status = redis_state.hgetall(key)
    assert status["state"] == "training"
    assert status["round"] == "2"
    assert status["epoch"] == "0"
    assert float(status["timestamp"]) > 0
    assert 0 < redis_state.ttl(key) <= 2 * state_manager.HEARTBEAT_INTERVAL


def test_availability_is_hospital_scoped_and_excludes_expired_nodes(redis_state, monkeypatch):
    monkeypatch.setenv("H_ID", "0")
    for h, n in [("0", "0"), ("0", "1"), ("1", "2")]:
        state_manager.update_node_status(h, n, "idle")
    assert set(state_manager.get_live_nodes("0")) == {"0", "1"}
    # Explicit expiry exercises Redis TTL behavior without a real-time sleep.
    redis_state.expire("fl:hospital:0:node:1", 0)
    assert orchestrator.select_k_nodes() == ["0"]
    redis_state.expire("fl:hospital:0:node:0", 0)
    assert orchestrator.select_k_nodes() == []


def test_status_transitions_to_error(redis_state):
    state_manager.update_node_status("0", "0", "training", 2, 1)
    state_manager.update_node_status("0", "0", "error", 2)
    assert redis_state.hget("fl:hospital:0:node:0", "state") == "error"


def test_twin_records_unique_epochs_and_elapsed_time(redis_state, monkeypatch):
    clock = Mock(side_effect=[10., 15., 20.])
    monkeypatch.setattr(state_manager, "time", SimpleNamespace(time=clock))
    for epoch in [0, 0, 1]:
        state_manager.record_twin_training("0", "0", 2, epoch)
    assert state_manager.get_twin_round("0", "0", 2) == {"epochs_seen": 2, "train_seconds": 10.}
    assert state_manager.get_twin_round("0", "0", 3) == {"epochs_seen": 0, "train_seconds": 0}


def test_heartbeat_listener_reads_nested_progress(redis_state, monkeypatch):
    monkeypatch.setenv("H_ID", "0")
    connection = Mock()
    connection.recv.return_value = json.dumps({"node_id": "1", "state": "training",
        "payload": {"round": 3, "epoch": 0}}).encode()
    listener = Mock()
    listener.accept.side_effect = [(connection, ("localhost", 1234)), OSError("end fixture")]
    monkeypatch.setattr(heartbeat_handler, "create_socket", Mock(return_value=listener))
    heartbeat_handler.listen()
    assert redis_state.hget("fl:hospital:0:node:1", "round") == "3"
    assert state_manager.get_twin_round("0", "1", 3)["epochs_seen"] == 1
    connection.close.assert_called_once()


@pytest.fixture
def scoring(redis_state, monkeypatch):
    monkeypatch.setattr(twin_trust, "EPOCHS", 4)
    monkeypatch.setattr(twin_trust, "load_holdout", lambda: (torch.zeros(4, 1), torch.tensor([0, 1, 0, 1])))
    # Small states encode controlled accuracy; isolate scoring from inference.
    monkeypatch.setattr(twin_trust, "holdout_accuracy", lambda sd: sd["w"].item())
    return {"w": torch.tensor(0.5)}


def update(node, accuracy=1., samples=100):
    return {"node_id": node, "state_dict": {"w": torch.tensor(accuracy)}, "num_samples": samples}


def observe(monkeypatch, records):
    monkeypatch.setattr(state_manager, "get_twin_round", lambda h, n, r: records[n])


def test_free_rider_excluded_and_historical_trust_decays(scoring, monkeypatch):
    observe(monkeypatch, {"0": {"epochs_seen": 0, "train_seconds": 0}})
    assert state_manager.get_trust("0", "0") == 1.
    for round_id, expected_trust in [(0, 0.7), (1, 0.49)]:
        updates, report = twin_trust.score_round("0", round_id, [update("0", accuracy=0.5)], scoring)
        assert updates[0]["weight"] == 0
        assert state_manager.get_trust("0", "0") == pytest.approx(expected_trust)
        assert "no_training" in report["0"]["flags"]


def test_unchanged_weights_rejected_despite_epoch_claims(scoring, monkeypatch):
    observe(monkeypatch, {"0": {"epochs_seen": 4, "train_seconds": 10}})
    updates, report = twin_trust.score_round("0", 0, [update("0", accuracy=0.5)], scoring)
    assert updates[0]["weight"] == 0
    assert "no_training" in report["0"]["flags"]


def test_inflated_samples_capped_against_observed_rate(scoring, monkeypatch):
    observe(monkeypatch, {n: {"epochs_seen": 4, "train_seconds": 10} for n in ["0", "1"]})
    updates, report = twin_trust.score_round("0", 0, [update("0"), update("1", samples=10000)], scoring)
    assert updates[0]["num_samples"] == updates[1]["num_samples"] == 100
    assert report["1"]["claimed_samples"] == 10000
    assert report["1"]["used_samples"] == 100
    assert "size_inflated" in report["1"]["flags"]
    assert "size_inflated" not in report["0"]["flags"]


def test_quality_and_prior_trust_control_fractional_weight(scoring, monkeypatch):
    observe(monkeypatch, {n: {"epochs_seen": 4, "train_seconds": 10} for n in ["0", "1"]})
    state_manager.set_trust("0", "1", 0.4)
    updates, report = twin_trust.score_round("0", 0, [update("0"), update("1", accuracy=0.75)], scoring)
    # chance=.5, best=1, quality=(.75-.5)/.5=.5; trust=.7*.4+.3*.5=.43.
    assert updates[1]["trust"] == pytest.approx(0.43)
    assert updates[1]["weight"] == pytest.approx(21.5)
    assert report["1"]["round_score"] == 0.5


def test_low_quality_model_excluded(scoring, monkeypatch):
    observe(monkeypatch, {n: {"epochs_seen": 4, "train_seconds": 10} for n in ["0", "1"]})
    updates, report = twin_trust.score_round("0", 0, [update("0"), update("1", accuracy=0.25)], scoring)
    assert updates[1]["weight"] == 0
    assert "low_quality" in report["1"]["flags"]


def test_holdout_inference_with_tiny_synthetic_model(monkeypatch):
    # Test the inference helper separately, using a deterministic linear layer.
    images = torch.eye(2).repeat(2, 1)
    labels = torch.tensor([0, 1, 0, 1])
    monkeypatch.setattr(twin_trust, "load_holdout", lambda: (images, labels))
    monkeypatch.setattr(twin_trust, "get_model", lambda: torch.nn.Linear(2, 2, bias=False))
    assert twin_trust.holdout_accuracy({"weight": torch.eye(2)}) == 1.
