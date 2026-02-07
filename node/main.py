from socket import *
import os
import json
import time
import threading
from datetime import datetime
from shared.config import HOSPITAL_HEARTBEAT_PORT, HEARTBEAT_INTERVAL, MODEL_CHUNK_SIZE
from shared.commons import create_socket
import uuid

# TODO: fix the setter functions

class NodeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = "idle"
        self._data = {}

    def get_status(self):
        with self._lock:
            return self._state, self._data.copy()

    def set_status(self, status, **data):
        with self._lock:
            self._state = status
            self._data = data

    def set_idle(self, **data):
        self.set_status("idle", **data)

    def set_training(self, epoch_num, round_num, **data):
        self.set_status("training", epoch=epoch_num, round=round_num, **data)

    def set_error(self, error_msg, **data):
        self.set_status("error", error_msg=error_msg, **data)

node_state = NodeState()

def send_status():
    hospital_hostname = os.getenv("H_HOST")
    node_id = os.getenv("NODE_ID", "unknown")

    state, data = node_state.get_status()
    status = {
        "node_id": node_id,
        "state": state,
        "timestamp": time.time(),
        "payload": data
    }
    
    try:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect((hospital_hostname, HOSPITAL_HEARTBEAT_PORT))
        sock.send(json.dumps(status).encode())
        sock.close()
        
    except Exception as e:
        print(f"Failed to send status: {e}")

def heartbeat_loop():
    while True:
        send_status()
        time.sleep(HEARTBEAT_INTERVAL)


def stream_model(path):
    model_id = uuid.uuid4()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(MODEL_CHUNK_SIZE)

            if not chunk:
                break
            yield chunk
'''
Edge server uploads the trained model to the hospital’s private blockchain network
Each bc_node:
receives (or fetches) the model
computes the hash
If hashes match across nodes:
model is verified
block is appended to Bi

why would they even differ?
'''

"""
model metadata must look like this:
metadata = {
    "model_id": acf059c8-217b-4e3c-a4c7-accbff22cd9d,
    "round": 5,
    "node_id": 2,
    "timestamp": time.time(),       # might be irrelevant
    "metrics": {
        "accuracy": 0.87.
        "loss": 0.23,
        "samples": 1000
    }
    "model_type": "resnet18",
}

headers = {
    'metadata': metadata,
    'Content-Type': 'application/octet-stream'
}
"""