from socket import *
import os
import json
import time
import threading
import asyncio
import argparse
from datetime import datetime
from shared.config import HOSPITAL_HEARTBEAT_PORT, HEARTBEAT_INTERVAL, MODEL_CHUNK_SIZE, BLOCKCHAIN_LEADER_HOSTNAME, BLOCKCHAIN_PORTS
from shared.commons import create_socket, get_logger
import uuid
import httpx

# TODO: fix the setter functions

logger = get_logger()

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


async def upload_model_to_blockchain(model_path, metadata):
    '''
    Edge server uploads the trained model DIRECTLY to the blockchain network.
    Each bc_node:
    - receives (or fetches) the model
    - computes the hash
    If hashes match across nodes:
    - model is verified
    - block is appended to Bi
    '''
    
    node_id = os.getenv("NODE_ID", "unknown")
    hospital_id = os.getenv("H_ID", "0")
    
    # Add required fields to metadata
    metadata["node_id"] = node_id
    metadata["hospital_id"] = f"hospital_{hospital_id}"
    metadata["model_id"] = str(uuid.uuid4())
    metadata["timestamp"] = time.time()
    
    bc_leader_url = f"http://{BLOCKCHAIN_LEADER_HOSTNAME}:{BLOCKCHAIN_PORTS['backend_port']}/submit_local_model"
    
    logger.info(f"Node {node_id}: Uploading model directly to blockchain leader at {bc_leader_url}")
    logger.info(f"Metadata: {metadata}")
    
    async def stream_file():
        with open(model_path, "rb") as f:
            while True:
                chunk = f.read(MODEL_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                bc_leader_url,
                headers={"metadata": json.dumps(metadata)},
                content=stream_file()
            )
            
            bc_response = response.json()
            logger.info(f"Blockchain response: {bc_response}")
            
            return bc_response
            
    except Exception as e:
        logger.error(f"Failed to upload model to blockchain: {e}")
        raise
'''
Edge server uploads the trained model to the hospital’s private blockchain network
Each bc_node:
receives (or fetches) the model
computes the hash
If hashes match across nodes:
model is verified
block is appended to Bi

why would they even differ? : Man in the middle attack
'''

"""
Model metadata must look like this:
metadata = {
    "model_id": "acf059c8-217b-4e3c-a4c7-accbff22cd9d",
    "round": 5,
    "node_id": 2,
    "hospital_id": "hospital_0",
    "timestamp": time.time(),
    "metrics": {
        "accuracy": 0.87,
        "loss": 0.23,
        "samples": 1000
    },
    "model_type": "resnet18",
}

headers = {
    'metadata': metadata,
    'Content-Type': 'application/octet-stream'
}
"""

def upload_model_sync(model_path, metadata=None):
    """
    Synchronous wrapper for upload_model_to_blockchain.
    Useful for testing and CLI usage.
    
    Args:
        model_path: Path to the model file
        metadata: Optional metadata dict. If None, creates default metadata.
    
    Returns:
        Blockchain response dict
    """
    if metadata is None:
        metadata = {
            "round": 0,
            "metrics": {
                "accuracy": 0.0,
                "loss": 0.0,
                "samples": 0
            },
            "model_type": "test_model"
        }
    
    return asyncio.run(upload_model_to_blockchain(model_path, metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Node Training Agent")
    parser.add_argument("--upload", type=str, help="Upload model file to blockchain")
    parser.add_argument("--round", type=int, default=0, help="Round number")
    parser.add_argument("--accuracy", type=float, default=0.0, help="Model accuracy")
    parser.add_argument("--loss", type=float, default=0.0, help="Model loss")
    parser.add_argument("--model-type", type=str, default="resnet18", help="Model type")
    
    args = parser.parse_args()
    
    if args.upload:
        logger.info(f"Testing model upload: {args.upload}")
        
        metadata = {
            "round": args.round,
            "metrics": {
                "accuracy": args.accuracy,
                "loss": args.loss,
                "samples": 1000
            },
            "model_type": args.model_type
        }
        
        try:
            response = upload_model_sync(args.upload, metadata)
            logger.info(f"Upload successful!")
            logger.info(f"Response: {json.dumps(response, indent=2)}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            exit(1)
    else:
        # Default: run heartbeat loop
        logger.info("Starting heartbeat loop...")
        heartbeat_loop()

def handle_training(epoch_num, round_num, node_state):
    # load_dataset()
    # load_global_model() - API

    # call the training function, pass the node_state to that function and set state in that function
    node_state.set_training(epoch_num, round_num)