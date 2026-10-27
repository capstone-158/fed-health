'''
systems stuff:
- Aggregation of local models received from blockchain
- Distribute global model back to edge servers

api:
- POST/receive_aggregated_model() - receive model from blockchain for aggregation
- GET/get_global_model() - edge servers retrieve aggregated global model

Fed FL Workflow:
- edge nodes train local models
- edge nodes upload DIRECTLY to blockchain
- blockchain verifies and stores model hashes
- central server (hospital) retrieves models from blockchain
- central server aggregates models (FedAvg, etc.)
- central server distributes global model back to edge servers
REPEAT FOR T ROUNDS
'''

# TODO: use the keys and implement tamperproof append

import torch
import io
import time
import uuid
import hashlib
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
import json
import os
from shared.config import BLOCKCHAIN_BACKEND_URL
from shared.commons import get_logger, get_model
try:
    from .crypto import *
except ImportError:
    from crypto import *

priv_key = None
pub_pem = None

logger = get_logger()
app = FastAPI()

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
# per hospital: all h_nodes mount the same ./h_node dir, and each hospital has its own model type
GLOBAL_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, f"hospital_{os.getenv('H_ID', '0')}_global_weights.pt")

CHUNK_SIZE = 4*1024*1024


def init_global_weights():
    # fresh GM_i for round 0. done once, so every node in the round starts from the same weights
    torch.save(get_model().state_dict(), GLOBAL_WEIGHTS_PATH)
    logger.info(f"Initialised global weights at {GLOBAL_WEIGHTS_PATH}")


# ---------------- talking to this hospital's private blockchain ----------------

def get_ledger_blocks():
    hospital_id = os.getenv("H_ID", "0")
    res = httpx.get(f"{BLOCKCHAIN_BACKEND_URL}/ledger/hospital_{hospital_id}", timeout=30)
    res.raise_for_status()
    chain = res.json()
    logger.info(f"Pulled hospital_{hospital_id} ledger from blockchain, {len(chain)} blocks")
    return [b for b in chain if b["payload"].get("type") != "GENESIS"]


def fetch_verified_state_dict(model_hash):
    # download the committed model and make sure it is exactly what the ledger recorded
    res = httpx.get(f"{BLOCKCHAIN_BACKEND_URL}/model/{model_hash}", timeout=300)
    res.raise_for_status()
    logger.info(f"Downloaded model {model_hash[:12]}... ({len(res.content)} bytes), checking hash against ledger")
    if hashlib.sha256(res.content).hexdigest() != model_hash:
        raise ValueError(f"Model {model_hash} does not match its ledger hash, tampered?")
    return torch.load(io.BytesIO(res.content), map_location="cpu")


def get_round_models(round_id, node_ids, round_started_at):
    '''
    Collect M_i of every node that trained in this round from B_i.
    returns [{"node_id", "state_dict", "num_samples"}, ...], num_samples is what the node claimed
    '''
    latest = {}
    for block in get_ledger_blocks():
        payload = block["payload"]
        # ledger keeps blocks of older runs too (maybe another data mode / model), only take this run's round
        if block["timestamp"] < round_started_at:
            continue
        if payload.get("kind", "local") == "local" and payload.get("round") == round_id \
                and payload.get("node_id") in node_ids:
            latest[payload["node_id"]] = payload    # later blocks win

    logger.info(f"Round {round_id}: found blocks on ledger for nodes {list(latest)}")
    missing = set(node_ids) - set(latest)
    if missing:
        logger.warning(f"Round {round_id}: no block on ledger for nodes {missing}, they wont be aggregated")

    client_updates = []
    for node_id, payload in latest.items():
        try:
            state_dict = fetch_verified_state_dict(payload["model_hash"])
            client_updates.append({"node_id": node_id, "state_dict": state_dict, "num_samples": payload["metrics"]["dataset_len"]})
            logger.info(f"Round {round_id}: hash matches, got verified model of node {node_id} ({payload['metrics']['dataset_len']} samples)")
        except Exception as e:
            logger.error(f"Round {round_id}: skipping model of node {node_id}: {e}")

    return client_updates


def submit_global_model(round_id):
    # record GM_i on B_i too (Alg. 4 verifies both M_i and GM_i)
    hospital_id = os.getenv("H_ID", "0")
    metadata = {
        "node_id": "global",
        "hospital_id": f"hospital_{hospital_id}",
        "model_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "round": round_id,
        "kind": "global",
    }
    logger.info(f"Sending round {round_id} global model to blockchain for verification, model_id: {metadata['model_id']}")
    with open(GLOBAL_WEIGHTS_PATH, "rb") as f:
        res = httpx.post(f"{BLOCKCHAIN_BACKEND_URL}/submit_local_model",
                         headers={"metadata": json.dumps(metadata)}, content=f.read(), timeout=300)
    res.raise_for_status()
    logger.info(f"Global model of round {round_id} recorded on blockchain: {res.json()}")


def record_trust(round_id, report):
    hospital_id = os.getenv("H_ID", "0")
    res = httpx.post(f"{BLOCKCHAIN_BACKEND_URL}/record_trust",
                     json={"hospital_id": f"hospital_{hospital_id}", "round": round_id, "nodes": report}, timeout=60)
    res.raise_for_status()
    logger.info(f"Twin trust: round {round_id} report recorded on blockchain: block {res.json()['block_index']}")


def get_latest_global_model_id():
    global_blocks = [b["payload"] for b in get_ledger_blocks() if b["payload"].get("kind") == "global"]
    return global_blocks[-1]["model_id"] if global_blocks else None

# crypto stuff below

@app.get("/get_certificate/{node_id}")
async def get_certificate(node_id: str):
    hospital_id = os.getenv("H_ID")

    cert_path = f"./cached/certs/node_{node_id}_cert.json"
    if os.path.exists(cert_path):
        import json
        with open(cert_path) as f:
            cert = json.load(f)
        return cert

    if priv_key is None:
        raise HTTPException(500, "Hospital not ready for key/cert generation yet.")

    cert = issue_node_certificate(hospital_id, node_id, priv_key)

    os.makedirs("./cached/certs", exist_ok=True)
    import json
    with open(cert_path, "w") as f:
        json.dump(cert, f, indent=2)
    
    return {"certificate": cert}

@app.get("/get_public_key")
async def get_public_key():
    pub_key_path = "./secrets/public_key.pem"

    if not os.path.exists(pub_key_path):
        raise HTTPException(404, "Public key not found.")
    
    with open(pub_key_path, "rb") as f:
        pub_key = f.read()

    return {
        "hospital_id": os.getenv("H_ID"),
        "public_key": pub_key
    }


def stream(path):
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            yield chunk


# now this can be changed to f.read()
@app.get("/get_global_weights")
def load_global_weights():
    # round 0: weights written by init_global_weights(); later rounds: overwritten by FedAvg
    if not os.path.exists(GLOBAL_WEIGHTS_PATH):
        raise HTTPException(503, f"{GLOBAL_WEIGHTS_PATH} not initialised yet")

    logger.info("NOW GONNA START STREAMING")
    return StreamingResponse(stream(GLOBAL_WEIGHTS_PATH), media_type="application/octet-stream")


@app.get("/get_model_type")
def get_model_type():
    return {
        "model": os.getenv("model_type"),
        "h_id": os.getenv("H_NODE_NAME")
    }