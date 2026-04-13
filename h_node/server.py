# this is where all the threads run
import time
import uvicorn
import threading
import logging
from heartbeat_handler import listen
from crypto import *    # custom module
from shared import state_manager
from shared.commons import *
import os
from cryptography.hazmat.primitives import serialization
from shared.config import CLIENTS, HOSPITAL_BACKEND_PORT
import orchestrator

logger = get_logger()

os.makedirs("./secrets", exist_ok=True)

private_key_path = "./secrets/private_key.pem"
public_key_path = "./secrets/public_key.pem"

if os.path.exists(private_key_path) and os.path.exists(public_key_path):
    with open(private_key_path, "rb") as f:
        priv_pem = f.read()
    with open(public_key_path, "rb") as f:
        pub_pem = f.read()
    
    priv_key = serialization.load_pem_private_key(priv_pem, password=None)
else:
    priv_key, priv_pem, pub_pem = generate_keys()
    
    with open(private_key_path, "wb") as f:
        f.write(priv_pem)
    
    with open(public_key_path, "wb") as f:
        f.write(pub_pem)
    
rounds_completed = 0

def precheck(default_mode_ref="global_round_0.pth"):
    redis_conn = state_manager.get_connection()
    hospital_id = os.getenv("H_ID")
    current_round = state_manager.get_hospital_current_round(hospital_id)
    return current_round


import backend
backend.priv_key = priv_key
backend.pub_pem = pub_pem

logger.info("Starting hospital server...")

heartbeat_thread = threading.Thread(target=listen, daemon=True)
heartbeat_thread.start()

def pub(round_id, node_ids, epochs=20, model_ref=None):
    r = state_manager.get_connection()

    hospital_id = os.getenv("H_ID")
    
    msg = {
        "type": "start training",
        "hospital_id": hospital_id,
        "round_id": int(round_id),
        "node_ids": node_ids,
        "epochs": int(epochs),
        "model_ref": model_ref,
        "timestamp": time.time()
    }
    payload = json.dumps(msg)
    receivers = r.publish("fl_commands", payload)
    logger.info(f"Published START_TRAINING round={round_id}, nodes={node_ids}, subscribers={receivers}")
    return {"published": True, "subscribers": receivers, "message": msg}

state_manager.get_connection()

# the function call below causes a problem because it runs the function before any heartbeats are sent
time.sleep(10)
while True:
    k_nodes = orchestrator.select_k_nodes()
    if k_nodes:
        if len(k_nodes) < CLIENTS:
            logging.warning(f"Only {len(k_nodes)} available, need {CLIENTS}. Using all")
        break
res = pub(0, k_nodes, 0, "global_round_1.pth")

while True:
    try:
        uvicorn.run(
            "backend:app",
            host="0.0.0.0",
            port=HOSPITAL_BACKEND_PORT,
            reload=False
        )
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        logger.info("Restarting in 5 seconds...")
        time.sleep(5)
