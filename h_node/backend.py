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
from shared.models import AdaptedAlexNet, AdaptedResNet18, ImprovedNet
from shared.state_manager import get_hospital_current_round
from shared.config import NUM_CLASSES
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
import json
import os
from shared.config import BLOCKCHAIN_LEADER_HOSTNAME, BLOCKCHAIN_PORTS, NODE_REGISTRY
from shared.commons import get_logger
from crypto import *

priv_key = None
pub_pem = None

logger = get_logger()
app = FastAPI()

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
GLOBAL_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "global_weights.pt")

CHUNK_SIZE = 4*1024*1024

def get_latest_model():
    hospital_id = os.getenv("H_ID", "0")
    ledger_path = f"bc_node/ledger/hospital_{hospital_id}_chain.json"

    with open(ledger_path, "r") as f:
        chain = json.load(f)

    latest_models = {}

    for block in chain:
        payload = block.get("payload", {})
        if payload.get("type") == "GENESIS":
            continue
        node_id = payload.get("node_id")
        if not node_id:
            continue
        latest_models[node_id] = {
            "model_id": payload["model_id"],
            "model_hash": payload["model_hash"],
            "metrics": payload.get("metrics", {})
        }
    return latest_models

# TODO: Implement model aggregation logic (FedAvg)
# TODO: Implement global model distribution to edge servers

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

# TODO: write an api for performing fedavg. api called by the orchestrator.


def stream(path):
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            yield chunk


# now this can be changed to f.read()
@app.get("/get_global_weights")
def load_global_weights():
    round_num = get_hospital_current_round()

    if round_num == 0:
        model = get_model()
        torch.save(model.state_dict(), GLOBAL_WEIGHTS_PATH)

    else:
        if not os.path.exists(GLOBAL_WEIGHTS_PATH):
            raise FileNotFoundError(f"{GLOBAL_WEIGHTS_PATH} not found")
        torch.load(GLOBAL_WEIGHTS_PATH, map_location="cpu")

    return StreamingResponse(stream(GLOBAL_WEIGHTS_PATH), media_type="application/octet-stream")


@app.get("get_model_type")
def get_model_type():
    return {
        "model": os.getenv("model_type"),
        "h_id": os.getenv("H_NODE_NAME")
    }