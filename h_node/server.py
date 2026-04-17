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
from shared.config import CLIENTS, HOSPITAL_BACKEND_PORT, HEARTBEAT_INTERVAL, ROUNDS
import orchestrator
import backend

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
    
backend.priv_key = priv_key
backend.pub_pem = pub_pem

def precheck():
    redis_conn = state_manager.get_connection()
    hospital_id = os.getenv("H_ID")
    current_round = state_manager.get_hospital_current_round(hospital_id)
    return current_round

completed_nodes = set()
lock = threading.Lock()
rounds_completed = 0

'''
TODO: bug fix
Bug 1 — listen_done doesn't filter by this hospital's nodes (line 69-75):


# current code — compares against the SENDER's hospital's round, not this h_node's
current_round = state_manager.get_hospital_current_round(data["hospital_id"])
if round_id != current_round:
    continue
with lock:
    completed_nodes.add(node_id)
All 3 h_nodes subscribe to fl_done. So h_node_0 receives done events from nodes 3, 4, 5 (hospitals 1 & 2) and counts them toward its own completed_nodes. 
The round appears complete when it's not.

fix:
# In listen_done — only count nodes belonging to THIS hospital
own_hospital_id = os.getenv("H_ID")
current_round = state_manager.get_hospital_current_round(own_hospital_id)
if round_id != current_round or data["hospital_id"] != own_hospital_id:
    continue
'''
def listen_done():
    r = state_manager.get_connection()
    pubsub = r.pubsub()
    pubsub.subscribe("fl_done")

    logger.info("Listening for done events...")

    for msg in pubsub.listen():
        if msg["type"] != "message":
            continue

        try:
            data = json.loads(msg["data"])
            round_id = data["round_id"]
            node_id = data["node_id"]

            current_round = state_manager.get_hospital_current_round(data["hospital_id"])

            if round_id != current_round:
                continue

            with lock:
                completed_nodes.add(node_id)
                logger.info(f"[ROUND {round_id}] Node {node_id} done. Completed nodes: {completed_nodes}")
        except Exception as e:
            logger.info(f"Some error in listen_done: {e}")


logger.info("Starting hospital server...")

heartbeat_thread = threading.Thread(target=listen, daemon=True)
heartbeat_thread.start()

def pub(round_id, node_ids, epochs=20, model_ref=None):
    r = state_manager.get_connection()

    hospital_id = os.getenv("H_ID")
    state_manager.set_hospital_current_round(hospital_id, round_id)
    state_manager.set_round_state(round_id, "started") 

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

def wait_for_round(round_id, hospital_id, timeout=600):
    start = time.time()

    while time.time() - start < timeout:
        with lock:
            if len(completed_nodes) >= CLIENTS:
                logger.info(f"Round {round_id} complete.")
                return True

        count = state_manager.barrier_count(round_id)
        if count >= CLIENTS:
            logger.info(f"Round {round_id} complete.")
            return True
        
        time.sleep(2)

    return False

def orchestration_loop():
    hospital_id = os.getenv("H_ID")

    round_id = state_manager.get_hospital_current_round(hospital_id)

    time.sleep(HEARTBEAT_INTERVAL)

    while True:
        logger.info(f"WE ARE CURRENTLY ON ROUND {round_id}")
        if round_id >= ROUNDS:
            logger.info(f"All {ROUNDS} rounds complete. FL Process ending!")
            break
        logger.info(f"Starting round {round_id}")

        k_nodes = orchestrator.select_k_nodes()

        if not k_nodes:
            logger.warning("No live nodes available")
            time.sleep(5)
            continue

        if len(k_nodes) < CLIENTS:
            logger.warning(f"Only {len(k_nodes)} nodes available")

        # reset local tracking
        with lock:
            completed_nodes.clear()

# TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!EVERYTHING CAN FAIL HERE DUE TO THE NAMING DIFFERENCE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        pub(round_id, node_ids=k_nodes, epochs=20,model_ref=f"global_round_{round_id}.pt")

        success = wait_for_round(round_id, hospital_id)

        if success:
            logger.info(f"Round {round_id} COMPLETE")

            state_manager.set_round_state(round_id, "completed")

            # TODO:
            # - receive client models
            # - validate model
            # - aggregate models
            # - update global model

            round_id += 1
            state_manager.set_hospital_current_round(hospital_id, round_id)

        else:
            logger.error(f"Round {round_id} FAILED")
            state_manager.set_round_state(round_id, "failed")

listen_done_thread = threading.Thread(target=listen_done, daemon=True)
listen_done_thread.start()

orchestration_thread = threading.Thread(target=orchestration_loop, daemon=True)
orchestration_thread.start()

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
