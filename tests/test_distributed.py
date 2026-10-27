"""Exercise real handlers with isolated storage and simulated network boundaries."""
import asyncio
import hashlib
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fakeredis
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def blockchain(tmp_path, monkeypatch):
    from bc_node import backend, ledger
    monkeypatch.setattr(ledger, "LEDGER_PATH", str(tmp_path / "ledgers"))
    monkeypatch.setattr(backend, "ledgers", {"hospital_0": ledger.Ledger("hospital_0")})
    monkeypatch.setattr(backend, "MODEL_STORE", str(tmp_path / "models"))
    monkeypatch.setattr(backend, "bc_nodes_registry", {})
    monkeypatch.setattr(backend, "replicate_block", Mock())
    return backend


@pytest.mark.parametrize("peer_hashes, expected", [
    (["match", "match"], "success"),
    (["match", "mismatch"], "consensus_failed"),
    (["match", None], "consensus_failed"),
    ([], "consensus_failed"),
])
def test_upload_consensus_commit_policy(blockchain, monkeypatch, peer_hashes, expected):
    content = b"synthetic model bytes"
    digest = hashlib.sha256(content).hexdigest()
    sockets = []
    registry = {f"peer-{i}": {"broadcast_port": 7000} for i in range(len(peer_hashes))}
    monkeypatch.setattr(blockchain, "bc_nodes_registry", registry)

    def send(node, info, chunks):
        received = bytearray()
        while (chunk := chunks.get()) is not None:
            received.extend(chunk)
        assert bytes(received) == content
        value = peer_hashes[int(node.rsplit("-", 1)[1])]
        if value is None:
            return None
        sock = Mock()
        sockets.append(sock)
        return (digest if value == "match" else "different-hash"), sock

    monkeypatch.setattr(blockchain, "send_to_node", send)
    with TestClient(blockchain.app) as client:
        response = client.post("/submit_local_model", content=content, headers={"metadata": json.dumps({
            "node_id": "0", "hospital_id": "hospital_0", "model_id": "m0", "round": 2,
        })})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == expected
    assert result["consensus"]["total"] == len(peer_hashes)
    assert result["consensus"]["verified"] == peer_hashes.count("match")
    success = expected == "success"
    assert len(blockchain.ledgers["hospital_0"].chain) == (2 if success else 1)
    from pathlib import Path
    committed = Path(blockchain.MODEL_STORE) / "commit" / f"{digest}.pt"
    assert committed.exists() == success
    if success:
        assert committed.read_bytes() == content
    assert not list((Path(blockchain.MODEL_STORE) / "stage").iterdir())
    for sock in sockets:
        sock.sendall.assert_called_once_with(b"COMMIT" if success else b"REJECT")
        sock.close.assert_called_once()


@pytest.mark.parametrize("tamper", [False, True])
def test_replication_handler_accepts_valid_rejects_tampered_blocks(blockchain, tamper):
    from bc_node.ledger import Block
    replica = blockchain.ledgers["hospital_0"]
    block = Block(1, replica.last_block().block_hash, {"model_hash": "original"})
    if tamper:
        block.payload["model_hash"] = "tampered"
    sock = Mock()
    sock.__enter__ = Mock(return_value=sock)
    sock.__exit__ = Mock(return_value=False)
    sock.recv.return_value = json.dumps({"type": "block_replication", "ledger_name": "hospital_0",
                                        "block": block.to_dict()}).encode()
    blockchain.handle_connection(sock, ("test-peer", 7000))
    sock.sendall.assert_called_once_with(b"ERROR" if tamper else b"OK")
    assert len(replica.chain) == (1 if tamper else 2)


@pytest.fixture
def edge(tmp_path, monkeypatch):
    monkeypatch.setenv("NODE_WORKDIR", str(tmp_path))
    monkeypatch.setenv("NODE_ID", "0")
    monkeypatch.setenv("H_ID", "0")
    module = importlib.import_module("node.main")
    monkeypatch.setattr(module, "download_weights", AsyncMock())
    monkeypatch.setattr(module, "client_update", Mock(return_value={"dataset_len": 10}))
    monkeypatch.setattr(module, "signal_round_complete", Mock())
    monkeypatch.setattr(module, "send_status", Mock())
    return module


@pytest.mark.parametrize("status", ["success", "consensus_failed", "waiting"])
def test_only_successful_upload_completes_round(edge, monkeypatch, status):
    monkeypatch.setattr(edge, "upload_model_to_blockchain", AsyncMock(return_value={"status": status}))
    state = edge.NodeState()
    if status == "success":
        asyncio.run(edge.handle_training(20, 3, state))
        edge.signal_round_complete.assert_called_once_with(3, "0", "0")
        assert state.get_status() == ("idle", {})
    else:
        with pytest.raises(RuntimeError, match="consensus"):
            asyncio.run(edge.handle_training(20, 3, state))
        edge.signal_round_complete.assert_not_called()
        assert state.get_status()[0] == "error"
        assert state.get_status()[1]["round"] == 3
        edge.send_status.assert_called_once()


@pytest.mark.parametrize("stage", ["download_weights", "client_update", "upload_model_to_blockchain"])
def test_round_exception_reports_failure_without_completion(edge, monkeypatch, stage):
    monkeypatch.setattr(edge, "upload_model_to_blockchain", AsyncMock(return_value={"status": "success"}))
    failing = AsyncMock(side_effect=RuntimeError("network or training failure")) if stage != "client_update" else Mock(side_effect=RuntimeError("network or training failure"))
    monkeypatch.setattr(edge, stage, failing)
    state = edge.NodeState()
    with pytest.raises(RuntimeError, match="network or training failure"):
        asyncio.run(edge.handle_training(20, 4, state))
    assert state.get_status()[0] == "error"
    assert state.get_status()[1]["round"] == 4
    edge.signal_round_complete.assert_not_called()


def test_node_state_returns_payload_copy(edge):
    state = edge.NodeState()
    state.set_training(2, 3)
    status, payload = state.get_status()
    payload["epoch"] = 99
    assert status == "training"
    assert state.get_status()[1]["epoch"] == 2


def test_command_listener_survives_failed_training(edge, monkeypatch):
    from node import server
    messages = [{"type": "message", "data": json.dumps({"type": "start training", "node_ids": ["0"],
                 "round_id": i, "epochs": 20})} for i in [1, 2]]
    redis = Mock()
    redis.pubsub.return_value.listen.return_value = iter(messages)
    training = AsyncMock(side_effect=[RuntimeError("failed round"), None])
    monkeypatch.setattr(server, "handle_training", training)
    server.listen("fl_commands", redis)
    assert training.await_count == 2


@pytest.fixture
def hospital(tmp_path, monkeypatch):
    # backend creates its weights directory on import; keep that in temporary storage.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("H_ID", "0")
    from h_node import server
    from shared import state_manager
    monkeypatch.setattr(state_manager, "r", fakeredis.FakeRedis(decode_responses=True))
    monkeypatch.setattr(server, "completed_nodes", set())
    return server


def test_round_completion_ignores_other_hospitals_and_old_rounds(hospital, monkeypatch):
    from shared import state_manager
    state_manager.set_hospital_current_round("0", 3)
    events = [("1", 3, "2"), ("0", 2, "old"), ("0", 3, "0"), ("0", 3, "0"), ("0", 3, "1")]
    redis = Mock()
    redis.pubsub.return_value.listen.return_value = iter([
        {"type": "message", "data": json.dumps({"hospital_id": h, "round_id": r, "node_id": n})}
        for h, r, n in events])
    monkeypatch.setattr(state_manager, "get_connection", lambda: redis)
    hospital.listen_done()
    assert hospital.completed_nodes == {"0", "1"}
    assert hospital.wait_for_round(3, "0", timeout=1)


def test_wait_for_round_times_out_without_enough_completions(hospital, monkeypatch):
    from shared import state_manager
    clock = Mock(side_effect=[0, 0, 3])
    monkeypatch.setattr(hospital, "time", SimpleNamespace(time=clock, sleep=Mock()))
    assert hospital.wait_for_round(3, "0", timeout=2) is False


def test_orchestrator_records_timeout_as_failed_round(hospital, monkeypatch):
    from shared import state_manager
    monkeypatch.setattr(hospital, "ROUNDS", 1)
    monkeypatch.setattr(hospital.backend, "init_global_weights", Mock())
    monkeypatch.setattr(hospital.orchestrator, "select_k_nodes", Mock(side_effect=[["0", "1"], RuntimeError("stop after failure")]))
    monkeypatch.setattr(hospital, "pub", Mock())
    monkeypatch.setattr(hospital, "wait_for_round", Mock(return_value=False))
    monkeypatch.setattr(hospital.time, "sleep", Mock())
    with pytest.raises(RuntimeError, match="stop after failure"):
        hospital.orchestration_loop()
    assert state_manager.get_round_state(0)["state"] == "failed"
    assert state_manager.get_hospital_current_round("0") == 0


def test_peer_still_pending_at_join_timeout_cannot_be_omitted_from_consensus(blockchain, monkeypatch):
    content = b"synthetic model"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(blockchain, "bc_nodes_registry", {"peer-0": {}, "peer-1": {}})
    sock = Mock()
    monkeypatch.setattr(blockchain, "send_to_node", lambda *args: (digest, sock))

    class PeerThread:
        def __init__(self, target, args, **kwargs):
            self.target, self.args = target, args

        def start(self):
            pass

        def join(self, timeout):
            # One thread completes, the other remains pending after the join timeout.
            if self.args[0] == "peer-0":
                self.target(*self.args)

    monkeypatch.setattr(blockchain, "threading", SimpleNamespace(Thread=PeerThread))

    class Request:
        headers = {"metadata": json.dumps({"node_id": "0", "hospital_id": "hospital_0", "round": 1})}

        async def stream(self):
            yield content

    result = asyncio.run(blockchain.submit_local_model(Request()))
    assert result["status"] == "consensus_failed"
    assert result["consensus"]["total"] == 2
    assert result["consensus"]["verified"] == 1
    sock.sendall.assert_called_once_with(b"REJECT")
    assert len(blockchain.ledgers["hospital_0"].chain) == 1


@pytest.mark.parametrize("status_code", [200, 503])
def test_upload_adapter_streams_model_and_rejects_http_errors(edge, tmp_path, monkeypatch, status_code):
    import httpx
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"synthetic weights")

    async def post(url, headers, content):
        data = b"".join([chunk async for chunk in content])
        assert data == model_path.read_bytes()
        metadata = json.loads(headers["metadata"])
        assert metadata["round"] == 7
        assert metadata["hospital_id"] == "hospital_0"
        assert metadata["node_id"] == "0"
        assert metadata["metrics"] == {"dataset_len": 10}
        return httpx.Response(status_code, json={"status": "success"}, request=httpx.Request("POST", url))

    client = Mock()
    client.post = AsyncMock(side_effect=post)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(edge.httpx, "AsyncClient", Mock(return_value=client))
    if status_code == 200:
        assert asyncio.run(edge.upload_model_to_blockchain(str(model_path), {"dataset_len": 10}, 7)) == {"status": "success"}
    else:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(edge.upload_model_to_blockchain(str(model_path), {"dataset_len": 10}, 7))


def test_six_edge_models_through_blockchain_fedavg_and_ensemble(tmp_path, monkeypatch):
    """In-process integration: real APIs/math/storage, simulated peer transport."""
    import importlib.util
    from pathlib import Path
    import threading
    from urllib.parse import urlparse
    import httpx
    import torch
    from bc_node import backend as leader, ledger
    from h_node import ml

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NODE_WORKDIR", str(tmp_path / "edge"))
    from node import main as edge
    from h_node import backend as hospital

    names = ["hospital_0", "hospital_1", "hospital_2", "multi_inst"]
    modules = [leader]
    # Separate backend modules give each peer its own state and ensemble handler.
    for i in [1, 2]:
        spec = importlib.util.spec_from_file_location(f"bc_node.smoke_peer_{i}", leader.__file__)
        peer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(peer)
        modules.append(peer)
    for i, module in enumerate(modules):
        monkeypatch.setattr(ledger, "LEDGER_PATH", str(tmp_path / f"replica-{i}"))
        monkeypatch.setattr(module, "ledgers", {name: ledger.Ledger(name) for name in names})
        monkeypatch.setattr(module, "MODEL_STORE", str(tmp_path / f"store-{i}"))
    monkeypatch.setattr(leader, "ensemble_submissions", {})
    monkeypatch.setattr(leader, "bc_nodes_registry", {
        f"peer-{i}": {"backend_url": f"http://peer-{i}", "broadcast_port": 7000}
        for i in [1, 2]})
    monkeypatch.setenv("BC_NODE_NAME", "leader")

    def verify_peer(node, info, chunks):
        peer = modules[int(node[-1])]
        content = bytearray()
        while (chunk := chunks.get()) is not None:
            content.extend(chunk)
        digest = hashlib.sha256(content).hexdigest()
        stage = Path(peer.MODEL_STORE) / "stage" / digest
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_bytes(content)

        class CommitSocket:
            def sendall(self, signal):
                if signal == b"COMMIT":
                    target = Path(peer.MODEL_STORE) / "commit" / f"{digest}.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stage.replace(target)
                else:
                    stage.unlink()

            def close(self):
                pass

        return digest, CommitSocket()

    def replicate(node, name, block):
        modules[int(node[-1])].ledgers[name].accept_block(ledger.Block.from_dict(block))

    threads = []

    class TrackedThread(threading.Thread):
        def start(self):
            threads.append(self)
            super().start()

    monkeypatch.setattr(leader, "send_to_node", verify_peer)
    monkeypatch.setattr(leader, "replicate_block", replicate)
    monkeypatch.setattr(leader, "threading", SimpleNamespace(Thread=TrackedThread))
    clients = {"bc_node_0": TestClient(leader.app), **{
        f"peer-{i}": TestClient(modules[i].app) for i in [1, 2]}}

    def route(method, url, **kwargs):
        parsed = urlparse(url)
        kwargs.pop("timeout", None)
        return clients[parsed.hostname].request(method, parsed.path, **kwargs)

    monkeypatch.setattr(httpx, "get", lambda url, **kw: route("GET", url, **kw))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: route("POST", url, **kw))
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async_client(
        transport=httpx.ASGITransport(app=leader.app), **kw))
    monkeypatch.setattr(ml, "get_model", lambda: torch.nn.Linear(2, 2, bias=False))
    data_path = tmp_path / "evaluation.pt"
    torch.save({"images": torch.eye(2), "labels": torch.tensor([0, 1])}, data_path)
    submissions = []
    for h in range(3):
        monkeypatch.setenv("H_ID", str(h))
        ids = [str(2 * h), str(2 * h + 1)]
        for j, node_id in enumerate(ids):
            monkeypatch.setenv("NODE_ID", node_id)
            path = tmp_path / f"edge-{node_id}.pt"
            torch.save({"weight": torch.eye(2) * (h + 1 + 2 * j)}, path)
            result = asyncio.run(edge.upload_model_to_blockchain(str(path), {"dataset_len": 1 + 2 * j}, 0))
            assert result["status"] == "success"
            assert result["consensus"] == {"all_same": True, "verified": 2, "total": 2}
        updates = hospital.get_round_models(0, ids, 0)
        assert len(updates) == 2
        weights_path = tmp_path / f"hospital-{h}.pt"
        averaged = ml.perform_fedavg([(u["state_dict"], u["num_samples"]) for u in updates], weights_path)
        assert torch.equal(averaged["weight"], torch.eye(2) * (h + 2.5))
        monkeypatch.setattr(hospital, "GLOBAL_WEIGHTS_PATH", str(weights_path))
        hospital.submit_global_model(0)
        model_id = hospital.get_latest_global_model_id()
        assert model_id is not None
        probs, labels = ml.predict_probs(str(weights_path), str(data_path))
        submissions.append({"hospital_id": f"hospital_{h}", "global_model_id": model_id,
                            "val": {"probs": probs, "labels": labels},
                            "test": {"probs": probs, "labels": labels}})
    responses = [clients["bc_node_0"].post("/submit_ensemble_probs", json=s).json() for s in submissions]
    assert [r["status"] for r in responses] == ["waiting", "waiting", "success"]
    result = responses[-1]["result"]
    assert result["val_accuracy"] == result["test_accuracy"] == 100
    assert result["global_model_ids"] == {s["hospital_id"]: s["global_model_id"] for s in submissions}
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    for name in names:
        expected = [b.to_dict() for b in leader.ledgers[name].chain]
        assert len(expected) == (2 if name == "multi_inst" else 4)
        for peer in modules[1:]:
            assert [b.to_dict() for b in peer.ledgers[name].chain] == expected
    for peer in modules:
        assert len(list((Path(peer.MODEL_STORE) / "commit").glob("*.pt"))) == 9
    for client in clients.values():
        client.close()
