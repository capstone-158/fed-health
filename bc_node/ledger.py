import time
import json
import hashlib
import threading
import os
from typing import Dict, List

LEDGER_PATH = os.getenv(
    "LEDGER_PATH", os.path.join(os.path.dirname(__file__), "ledger")
)

class Block:
    def __init__(self, index, prev_hash, payload, timestamp=None, block_hash=None):
        self.index = index
        self.prev_hash = prev_hash
        self.timestamp = timestamp or time.time()
        self.payload = payload
        self.block_hash = block_hash or self.compute_hash()

    def compute_hash(self):
        """
        According to paper: block_hash = SHA256(index + prev_hash + payload)
        """
        block_data = json.dumps(
            {
                "index": self.index,
                "prev_hash": self.prev_hash,
                "timestamp": self.timestamp,
                "payload": self.payload
            },
            sort_keys=True
        )
        return hashlib.sha256(block_data.encode()).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "block_hash": self.block_hash
        }

    @staticmethod
    def from_dict(d):
        return Block(
            index=d["index"],
            prev_hash=d["prev_hash"],
            timestamp=d["timestamp"],
            payload=d["payload"],
            block_hash=d["block_hash"]
        )

class Ledger:
    def __init__(self, ledger_name):
        self.ledger_name = ledger_name
        self.lock = threading.Lock()
        os.makedirs(LEDGER_PATH, exist_ok=True)
        self.file_path = os.path.join(
            LEDGER_PATH, f"{ledger_name}_chain.json"
        )
        self.chain: List[Block] = []
        self._load()

    def _create_genesis_block(self):
        if len(self.chain) != 0:
            return
        genesis_block = Block(
            index=0,
            prev_hash="0",
            payload={"type": "GENESIS"}
        )
        self.chain.append(genesis_block)
        self._persist()

    def _load(self):
        if not os.path.exists(self.file_path):
            self._create_genesis_block()
            return
        with open(self.file_path, "r") as f:
            data = json.load(f)
        self.chain = []
        for b in data:
            blk = Block.from_dict(b)
            self.chain.append(blk)
        if len(self.chain) == 0:
            self._create_genesis_block()

    def _persist(self):
        blocks = []
        for b in self.chain:
            blocks.append(b.to_dict())
        with open(self.file_path, "w") as f:
            json.dump(blocks, f, indent=2)

    def last_block(self):
        if len(self.chain) == 0:
            return None
        return self.chain[-1]

    def append_block(self, payload):
        """
        Append a block to the chain.
        For private blockchain (hospital_X): payload should contain
            - hospital_id, model_hash, model_id, metrics, trust_score
        For multi-institutional blockchain: payload should contain
            - set_of_model_id, set_of_weights, accuracy
        """
        with self.lock:
            last_block = self.last_block()
            new_block = Block(
                index=last_block.index + 1,
                prev_hash=last_block.block_hash,
                payload=payload
            )
            self.chain.append(new_block)
            self._persist()
            return new_block

ledgers: Dict[str, Ledger] = {
    "hospital_0": Ledger("hospital_0"),
    "hospital_1": Ledger("hospital_1"),
    "hospital_2": Ledger("hospital_2"),
    "multi_inst": Ledger("multi_inst")
}