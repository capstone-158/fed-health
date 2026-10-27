import os
import json
import pytest
from bc_node.ledger import Block, Ledger


def test_genesis_block_created(tmp_path, ledger_dir):
    ledger = Ledger(f"test_{tmp_path.name}")
    assert len(ledger.chain) == 1
    assert ledger.chain[0].payload == {"type": "GENESIS"}
    assert ledger.chain[0].prev_hash == "0"


def test_blocks_are_linked(tmp_path, ledger_dir):
    ledger = Ledger(f"test_{tmp_path.name}")
    b1 = ledger.append_block({"model_hash": "abc"})
    b2 = ledger.append_block({"model_hash": "def"})
    assert b1.prev_hash == ledger.chain[0].block_hash
    assert b2.prev_hash == b1.block_hash
    assert b2.index == 2


def test_hash_is_deterministic():
    b = Block(1, "prev", {"x": 1}, timestamp=123.0)
    assert b.compute_hash() == Block(1, "prev", {"x": 1}, timestamp=123.0).compute_hash()


def test_payload_change_invalidates_hash():
    b = Block(1, "prev", {"model_hash": "abc"})
    b.payload["model_hash"] = "evil"
    assert b.compute_hash() != b.block_hash


def test_ledger_persists_to_disk(tmp_path, ledger_dir):
    name = f"test_{tmp_path.name}"
    original = Ledger(name)
    original.append_block({"model_hash": "abc"})
    before = [b.to_dict() for b in original.chain]
    reloaded = Ledger(name)
    assert [b.to_dict() for b in reloaded.chain] == before
    assert len(reloaded.chain) == 2
    assert reloaded.chain[1].payload == {"model_hash": "abc"}
    assert os.path.exists(reloaded.file_path)


def test_independent_replicas_accept_identical_blocks(ledger_dir, monkeypatch):
    from bc_node import ledger
    replicas = []
    for i in range(3):
        monkeypatch.setattr(ledger, "LEDGER_PATH", str(ledger_dir / str(i)))
        replicas.append(Ledger("hospital_0"))
    assert len({r.chain[0].block_hash for r in replicas}) == 1
    for payload in [{"model_hash": "abc"}, {"kind": "global", "round": 1}]:
        block = replicas[0].append_block(payload)
        for replica in replicas[1:]:
            replica.accept_block(Block.from_dict(block.to_dict()))
    assert all([b.to_dict() for b in r.chain] == [b.to_dict() for b in replicas[0].chain] for r in replicas)


@pytest.mark.parametrize("change", ["payload", "timestamp", "hash", "empty_hash", "null_hash", "link", "index"])
def test_modified_disk_chain_rejected(ledger_dir, change):
    ledger = Ledger("tamper")
    ledger.append_block({"model_hash": "original"})
    data = [b.to_dict() for b in ledger.chain]
    if change == "payload": data[1]["payload"]["model_hash"] = "modified"
    elif change == "timestamp": data[1]["timestamp"] += 1
    elif change == "hash": data[1]["block_hash"] = "forged"
    elif change == "empty_hash": data[1]["block_hash"] = ""
    elif change == "null_hash": data[1]["block_hash"] = None
    else:
        # Even a recomputed hash must not bypass linkage/index validation.
        data[1]["prev_hash" if change == "link" else "index"] = "wrong" if change == "link" else 9
        block = Block.from_dict(data[1])
        data[1]["block_hash"] = block.compute_hash()
    with open(ledger.file_path, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="Invalid block"):
        Ledger("tamper")


def test_rejected_replication_preserves_memory_and_disk(ledger_dir):
    ledger = Ledger("replica")
    before = open(ledger.file_path).read()
    block = Block(1, ledger.last_block().block_hash, {"model_hash": "abc"})
    block.payload["model_hash"] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        ledger.accept_block(block)
    assert len(ledger.chain) == 1
    assert open(ledger.file_path).read() == before


def test_legacy_valid_timestamped_genesis_still_loads(ledger_dir):
    block = Block(0, "0", {"type": "GENESIS"}, timestamp=123.0)
    (ledger_dir / "legacy_chain.json").write_text(json.dumps([block.to_dict()]))
    assert Ledger("legacy").chain[0].to_dict() == block.to_dict()
