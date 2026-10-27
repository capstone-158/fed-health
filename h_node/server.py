# this is where all the threads run
import time
import torch
import uvicorn
import threading
import logging
from heartbeat_handler import listen
from crypto import *    # custom module
from shared import state_manager
from shared.commons import *
import os
from cryptography.hazmat.primitives import serialization
from shared.config import CLIENTS, HOSPITAL_BACKEND_PORT, HEARTBEAT_INTERVAL, ROUNDS, EPOCHS, ROUND_TIMEOUT, BLOCKCHAIN_BACKEND_URL, VAL_DATA_PATH, TEST_DATA_PATH, TWIN_TRUST
import orchestrator
import backend
import httpx
from ml import perform_fedavg, predict_probs
from shared.ensemble import format_summary
import twin_trust

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

            # All 3 h_nodes subscribe to the same `fl_done` channel, so count only
            # done-events from THIS hospital's own nodes against THIS hospital's
            # current round. Otherwise other hospitals' nodes inflate the count and
            # the round appears complete before it actually is.
            own_hospital_id = os.getenv("H_ID")
            current_round = state_manager.get_hospital_current_round(own_hospital_id)

            if round_id != current_round or data["hospital_id"] != own_hospital_id:
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

def wait_for_round(round_id, hospital_id, timeout=ROUND_TIMEOUT):
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

    if round_id == 0 or not os.path.exists(backend.GLOBAL_WEIGHTS_PATH):
        backend.init_global_weights()

    time.sleep(HEARTBEAT_INTERVAL)

    last_no_nodes_warning = 0

    while True:
        if round_id >= ROUNDS:
            logger.info(f"All {ROUNDS} rounds complete. FL Process ending!")
            break

        k_nodes = orchestrator.select_k_nodes()

        if not k_nodes:
            # keep waiting for nodes, but dont flood the logs every 5s
            if time.time() - last_no_nodes_warning > 60:
                logger.warning(f"No live nodes available for round {round_id}, waiting for heartbeats...")
                last_no_nodes_warning = time.time()
            time.sleep(5)
            continue

        logger.info(f"WE ARE CURRENTLY ON ROUND {round_id}")
        logger.info(f"Starting round {round_id}, waiting up to {ROUND_TIMEOUT}s for nodes {k_nodes}")

        if len(k_nodes) < CLIENTS:
            logger.warning(f"Only {len(k_nodes)} nodes available")

        # reset local tracking
        with lock:
            completed_nodes.clear()

# TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!EVERYTHING CAN FAIL HERE DUE TO THE NAMING DIFFERENCE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        round_started_at = time.time()
        pub(round_id, node_ids=k_nodes, epochs=EPOCHS,model_ref=f"global_round_{round_id}.pt")

        success = wait_for_round(round_id, hospital_id)

        if success:
            logger.info(f"Round {round_id} COMPLETE")

            state_manager.set_round_state(round_id, "completed")

            # collect this round's local models from B_i, FedAvg them into GM_i
            logger.info(f"Aggregation: pulling round {round_id} local models of nodes {k_nodes} from blockchain")
            client_updates = backend.get_round_models(round_id, k_nodes, round_started_at)
            for u in client_updates:
                u["weight"] = u["num_samples"]

            if client_updates and TWIN_TRUST:
                global_state = torch.load(backend.GLOBAL_WEIGHTS_PATH, map_location="cpu")
                client_updates, report = twin_trust.score_round(hospital_id, round_id, client_updates, global_state)
                try:
                    backend.record_trust(round_id, report)
                except Exception as e:
                    logger.error(f"Twin trust: could not record round {round_id} report on blockchain: {e}")
                client_updates = [u for u in client_updates if u["weight"] > 0]

            if client_updates:
                logger.info(f"Aggregation: running fedavg on {len(client_updates)} models, weight per node: { {u['node_id']: round(u['weight'], 2) for u in client_updates} }")
                perform_fedavg([(u["state_dict"], u["weight"]) for u in client_updates], backend.GLOBAL_WEIGHTS_PATH)
                logger.info(f"Aggregation: round {round_id} global model saved to {backend.GLOBAL_WEIGHTS_PATH}, nodes get this next round")
                try:
                    backend.submit_global_model(round_id)
                except Exception as e:
                    logger.error(f"Aggregation: could not record round {round_id} global model on blockchain: {e}")
            else:
                logger.error(f"Aggregation: no usable local models for round {round_id}, keeping previous global model")

            round_id += 1
            state_manager.set_hospital_current_round(hospital_id, round_id)

        else:
            logger.error(f"Round {round_id} FAILED")
            state_manager.set_round_state(round_id, "failed")

    logger.info("!!!!!!!!!!LOCAL FL DONE, MOVING TO MULTI-INSTITUTIONAL ENSEMBLE!!!!!!!!!!")
    submit_for_ensemble()


def submit_for_ensemble():
    '''
    After local FL is done, send P_Hi of the final GM_i on the shared val/test splits to
    the multi-institutional blockchain B_M, which tunes the ensemble weights (Alg. 5-6).
    '''
    hospital_id = os.getenv("H_ID")
    logger.info(f"Ensemble: running final global model {backend.GLOBAL_WEIGHTS_PATH} on val split {VAL_DATA_PATH}")
    val_probs, val_labels = predict_probs(backend.GLOBAL_WEIGHTS_PATH, VAL_DATA_PATH)
    logger.info(f"Ensemble: running final global model {backend.GLOBAL_WEIGHTS_PATH} on test split {TEST_DATA_PATH}")
    test_probs, test_labels = predict_probs(backend.GLOBAL_WEIGHTS_PATH, TEST_DATA_PATH)

    global_model_id = backend.get_latest_global_model_id()
    logger.info(f"Ensemble: latest global model id on blockchain is {global_model_id}")

    submission = {
        "hospital_id": f"hospital_{hospital_id}",
        "global_model_id": global_model_id,
        "val": {"probs": val_probs, "labels": val_labels},
        "test": {"probs": test_probs, "labels": test_labels},
    }
    logger.info(f"Ensemble: sending {len(val_probs)} val + {len(test_probs)} test probabilities to blockchain leader at {BLOCKCHAIN_BACKEND_URL}/submit_ensemble_probs")
    try:
        res = httpx.post(f"{BLOCKCHAIN_BACKEND_URL}/submit_ensemble_probs", json=submission, timeout=600)
        res.raise_for_status()
        bc_response = res.json()
        if bc_response["status"] == "waiting":
            logger.info(f"Ensemble: submitted, leader is still waiting on other hospitals. received so far: {bc_response['received']}")
        elif bc_response["status"] == "success":
            logger.info(f"!!!!!!!!!!ENSEMBLE WEIGHTS ON MULTI-INST BLOCKCHAIN!!!!!!!!!! block {bc_response['block_index']}")
            logger.info(format_summary(bc_response["result"], bc_response["block_index"]))
        else:
            logger.error(f"Ensemble: blockchain could not agree on the weights: {bc_response}")
        logger.info(f"Ensemble: hospital_{hospital_id} done. final weights can be read from {BLOCKCHAIN_BACKEND_URL}/ensemble_result")
    except Exception as e:
        logger.error(f"Ensemble: could not submit probabilities for ensemble tuning: {e}")

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
