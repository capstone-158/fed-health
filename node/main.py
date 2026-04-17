from socket import *
import uuid
import requests
import torch
import os
import json
import time
import threading
import asyncio
import argparse
from datetime import datetime
from shared.config import (
    HOSPITAL_HEARTBEAT_PORT,
    HOSPITAL_BACKEND_PORT,
    HEARTBEAT_INTERVAL,
    MODEL_CHUNK_SIZE,
    BLOCKCHAIN_LEADER_HOSTNAME,
    HOSPITAL_URL,
    BLOCKCHAIN_PORTS,
    BLOCKCHAIN_BACKEND_URL
)
from shared.commons import create_socket, get_logger
from shared.state_manager import get_connection
import uuid
import httpx
from training.ml_utils import client_update

# TODO: fix the setter functions
logger = get_logger()

# TODO: call submit_model API in the flow

DATA_PATH = "/app/shared/tiny_dataset.pt"
NODE_ID_ENV = os.getenv("NODE_ID", "unknown")
WEIGHTS_PATH = f"/app/global_weights/weights_{NODE_ID_ENV}.pt"
MODEL_PATH = f"/app/models/model_{NODE_ID_ENV}.pt"

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
        "payload": data,
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


async def upload_model_to_blockchain(model_path, model_metrics):
    """
    Edge server uploads the trained model DIRECTLY to the blockchain network.
    Each bc_node:
    - receives (or fetches) the model
    - computes the hash
    If hashes match across nodes:
    - model is verified
    - block is appended to Bi
    """

    node_id = os.getenv("NODE_ID", "unknown")
    hospital_id = os.getenv("H_ID", "0")

    metadata = {}

    # Add required fields to metadata
    metadata["node_id"] = node_id
    metadata["hospital_id"] = f"hospital_{hospital_id}"
    metadata["model_id"] = str(uuid.uuid4())
    metadata["timestamp"] = time.time()
    metadata["metrics"] = model_metrics

    bc_leader_url = f"http://{BLOCKCHAIN_LEADER_HOSTNAME}:{BLOCKCHAIN_PORTS['backend_port']}/submit_local_model"

    logger.info(
        f"Node {node_id}: Uploading model directly to blockchain leader at {bc_leader_url}"
    )
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
                content=stream_file(),
            )

            bc_response = response.json()
            logger.info(f"Blockchain response: {bc_response}")

            return bc_response

    except Exception as e:
        logger.error(f"Failed to upload model to blockchain: {e}")
        raise


"""
Edge server uploads the trained model to the hospital’s private blockchain network
Each bc_node:
receives (or fetches) the model
computes the hash
If hashes match across nodes:
model is verified
block is appended to Bi

why would they even differ? : Man in the middle attack
"""

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


# this function is bs
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
            "metrics": {"accuracy": 0.0, "loss": 0.0, "samples": 0},
            "model_type": "test_model",
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
            "metrics": {"accuracy": args.accuracy, "loss": args.loss, "samples": 1000},
            "model_type": args.model_type,
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

async def download_weights(url, target_path):
    temp_file_path = f"weights_temp_{uuid.uuid4()}.pt"
    async with httpx.AsyncClient() as client:
        async with client.stream('GET', url) as res:
            res.raise_for_status()
            with open(temp_file_path, "wb") as f:
                async for chunk in res.aiter_bytes():
                    f.write(chunk)
            logger.info(f"saved model to {temp_file_path}")

    os.replace(temp_file_path, target_path)

def signal_round_complete(round_id, node_id, hospital_id):
    channel = f"fl_done"

    r = get_connection()

    r.publish(channel, json.dumps({
        "round_id": round_id,
        "node_id": node_id,
        "hospital_id": hospital_id
    }))

    logger.info("Sent barrier checking to hospital")

async def handle_training(epoch_num, round_num, node_state):
    '''
        currently implemented:
        - loading the dataset
        - getting global weights from the hospital
        - training the global model on the dataset
        - submitting the model to the blockchain for verification
        
        next up:
        - where do u save the model after consensus
        - how does the hospital get the model
        - send "done" signal in redis pub/sub between node and hospital
    '''
    
    # loading the global model
    endpoint = "/get_global_weights"
    await download_weights(HOSPITAL_URL + endpoint, WEIGHTS_PATH)
    logger.info("Training loop: Downloaded weights")

    # training model
    logger.info("Training loop: Starting training now")
    node_state.set_training(epoch_num, round_num)
    metrics = client_update(DATA_PATH, WEIGHTS_PATH, MODEL_PATH)
    node_state.set_idle()

    # submitting model to blockchain
    logger.info("Sending model to blockchain for verification")
    res = await upload_model_to_blockchain(MODEL_PATH, metrics)
    logger.info(f"Response received from blockchain: {res}")

    # barrier checkin
    signal_round_complete(round_num, os.getenv("NODE_ID"), os.getenv("H_ID"))