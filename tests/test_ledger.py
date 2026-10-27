import os
from bc_node.ledger import Block, Ledger


def test_genesis_block_created(tmp_path):
    ledger = Ledger(f"test_{tmp_path.name}")
    assert len(ledger.chain) == 1
    assert ledger.chain[0].payload == {"type": "GENESIS"}
    assert ledger.chain[0].prev_hash == "0"


def test_blocks_are_linked(tmp_path):
    ledger = Ledger(f"test_{tmp_path.name}")
    b1 = ledger.append_block({"model_hash": "abc"})
    b2 = ledger.append_block({"model_hash": "def"})
    assert b1.prev_hash == ledger.chain[0].block_hash
    assert b2.prev_hash == b1.block_hash
    assert b2.index == 2


def test_hash_is_deterministic():
    b = Block(1, "prev", {"x": 1}, timestamp=123.0)
    assert b.compute_hash() == Block(1, "prev", {"x": 1}, timestamp=123.0).compute_hash()


def test_tampered_payload_detected():
    b = Block(1, "prev", {"model_hash": "abc"})
    b.payload["model_hash"] = "evil"
    assert b.compute_hash() != b.block_hash


def test_ledger_persists_to_disk(tmp_path):
    name = f"test_{tmp_path.name}"
    Ledger(name).append_block({"model_hash": "abc"})
    reloaded = Ledger(name)
    assert len(reloaded.chain) == 2
    assert reloaded.chain[1].payload == {"model_hash": "abc"}
    assert os.path.exists(reloaded.file_path)
