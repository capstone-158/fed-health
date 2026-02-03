'''
    how do we share the blockchain and keep it sync everywhere?
    how do we broadcast? DIGITAL TWIN? redis
    how to prevent faking of updates to other's ledgers?
'''

'''
    Broadcast mecahnism:
        - local node trains a model
        - node generates a hash, and a uuid
        -  broadcast to all the nodes
    - perform concensus and if verified, append to the ledger

Logical implementation of 2 different blockchains into one single network:
    Physical blockchain network (3 bc_nodes)
    │
    ├── Ledger: hospital_0_chain   (private to Hospital 0)
    ├── Ledger: hospital_1_chain   (private to Hospital 1)
    ├── Ledger: hospital_2_chain   (private to Hospital 2)
    └── Ledger: multi_inst_chain  (shared, ensemble layer)

Block Structure:
    Block {
    index: int
    prev_hash: str
    timestamp: float
    payload: {
        hospital_id
        model_hash
        metrics
        trust_score
    }
    block_hash: str
    }

block_hash = SHA256(index + prev_hash + payload)

'''

'''
    bc_node to bc_node: communication happens through TCP sockets.
'''

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
    def __init__(self, index, prev_hash, payload, nonce, timestamp=None, block_hash=None):
        self.index = index
        self.prev_hash = prev_hash
        self.timestamp = timestamp or time.time()
        self.payload = payload
        self.nonce = nonce
        self.block_hash = block_hash or self.compute_hash()

    def compute_hash(self):
        block_data = json.dumps(
            {
                "index": self.index,
                "prev_hash": self.prev_hash,
                "timestamp": self.timestamp,
                "payload": self.payload,
                "nonce": self.nonce
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
            "nonce": self.nonce,
            "block_hash": self.block_hash
        }

    @staticmethod
    def from_dict(d):
        return Block(
            index=d["index"],
            prev_hash=d["prev_hash"],
            timestamp=d["timestamp"],
            payload=d["payload"],
            nonce=d["nonce"],
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
            payload={"type": "GENESIS"},
            nonce=0
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
        with self.lock:
            last_block = self.last_block()
            new_block = Block(
                index=last_block.index + 1,
                prev_hash=last_block.block_hash,
                payload=payload,
                nonce=0
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