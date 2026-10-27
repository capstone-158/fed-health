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

import shutil
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
import httpx
import json
import hashlib
import os
import queue
from socket import *
from shared.config import BLOCKCHAIN_PORTS, BLOCKCHAIN_LEADER_HOSTNAME, MODEL_CHUNK_SIZE, NUM_HOSPITALS
from shared.ensemble import tune_ensemble, result_hash, format_summary
from shared.commons import *
import time
from ping3 import ping
import threading
try:
    from .ledger import ledgers, Block
except ImportError:  # direct server.py launch inside the container
    from ledger import ledgers, Block

logger = get_logger()
app = FastAPI()
MODEL_STORE = os.getenv("MODEL_STORE", "/app/client_models")

# os.makedirs("/models")

bc_nodes_registry = {}

def register_nodes():
    try:
        sock = create_socket("0.0.0.0", BLOCKCHAIN_PORTS["leader_checkin_port"])
        sock.listen()

        logger.info(f"Leader listening for chekcin on port {BLOCKCHAIN_PORTS['leader_checkin_port']}")

        while True:
            client_sock, addr = sock.accept()

            with client_sock:
                try:
                    data = client_sock.recv(4096).decode().strip()
                    msg = json.loads(data)

                    logger.info(f"Checkin msg received {msg}")

                    host = msg["hostname"]
                    bc_nodes_registry[host] = {
                        "broadcast_port": msg["broadcast_port"],
                        "backend_url": msg["backend_url"]
                    }

                    client_sock.sendall(b"OK")
                except Exception as e:
                    logger.error(f"Error handling check-in from {addr}: {e}")
                    client_sock.sendall(b"ERROR")

    except:
        logger.error("Failed to open socket for checkins. Leader failure.")


# TODO: change to heartbeats if needed
def checkin():
    time.sleep(2)
    hostname = os.getenv("BC_NODE_NAME")
    if not hostname:
        logger.error("No env vars for hostname")
        exit(1)

    checkin_msg = {
        "hostname": hostname,
        "broadcast_port": BLOCKCHAIN_PORTS["broadcast_port"],
        "backend_url": f"http://{hostname}:{BLOCKCHAIN_PORTS['backend_port']}"
    }
    
    try:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.settimeout(10) 
        sock.connect((BLOCKCHAIN_LEADER_HOSTNAME, int(BLOCKCHAIN_PORTS["leader_checkin_port"])))

        while True:
            sock.sendall((json.dumps(checkin_msg) + '\n').encode())

            ack = sock.recv(1024).decode().strip()
            if ack != "OK":
                logger.error("Checkin Failed.")
            else:
                break

    except Exception as e:
        logger.error(f"Error in sending check-in msg. Error: {e}")

def send_to_node(node, info, chunk_queue):
    try:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.settimeout(60)
        sock.connect((node, info["broadcast_port"]))

        metadata = {
            "type": "model_broadcast",
            "node": node,
            "info": info
        }
        sock.sendall(json.dumps(metadata).encode())

        ack = sock.recv(1024)
        if ack != b"OK":
            logger.error(f"Node {node} not ready: {ack}")
            sock.close()
            return None

        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            sock.sendall(chunk)

        sock.sendall(b"eof")

        response_hash = sock.recv(1024).decode()
        logger.info(f"Received hash from {node}: {response_hash}")

        return response_hash, sock

    except Exception as e:
        logger.error(f"Error in sending chunk to node. Error: {e}")
        return None


def recv_model_and_hash():
    try:
        sock = create_socket("0.0.0.0", BLOCKCHAIN_PORTS["broadcast_port"])
        sock.listen()
        
        logger.info(f"Listening for model broadcasts on port {BLOCKCHAIN_PORTS['broadcast_port']}")

        os.makedirs(f"{MODEL_STORE}/stage", exist_ok=True)
        os.makedirs(f"{MODEL_STORE}/commit", exist_ok=True)

        while True:
            client_sock, addr = sock.accept()
            # one thread per connection: with a single handler, two uploads arriving at the same time
            # could be picked up in opposite order by the two peers and deadlock each other
            threading.Thread(target=handle_connection, args=(client_sock, addr), daemon=True).start()

    except Exception as e:
        logger.error(f"Failed to start model receiver: {e}")


def handle_connection(client_sock, addr):
    with client_sock:
        try:
            metadata_raw = client_sock.recv(4096).decode()
            metadata = json.loads(metadata_raw)

            logger.info(f"Received metadata from {addr}: {metadata}")

            if metadata.get("type") == "model_broadcast":
                client_sock.sendall(b"OK")
                logger.info(f"sent ok to {addr}, ready to receive model")

                hasher = hashlib.sha256()
                stage_path = f"{MODEL_STORE}/stage/model_{time.time()}.pt"

                with open(stage_path, "wb") as f:
                    while True:
                        chunk = client_sock.recv(MODEL_CHUNK_SIZE)
                        if not chunk:
                            break

                        if chunk.endswith(b"eof"):
                            if len(chunk) > 3:
                                hasher.update(chunk[:-3])
                                f.write(chunk[:-3])
                            break

                        hasher.update(chunk)
                        f.write(chunk)

                    model_hash = hasher.hexdigest()
                    logger.info(f"received model from {addr}, hash: {model_hash}")
                    logger.info(f"Model: {model_hash} STAGED")

                    client_sock.sendall(model_hash.encode())

                final_path = f"{MODEL_STORE}/commit/{model_hash}.pt"
                # listen for commit signal then save the model to a final path
                signal = client_sock.recv(16).decode()
                if signal == "COMMIT":
                    os.replace(stage_path, final_path)
                    logger.info(f"Model: {model_hash} COMMITTED")
                else:
                    os.remove(stage_path)
                    logger.info(f"Model: {model_hash} REJECTED by leader, removed from stage")

            elif metadata.get("type") == "block_replication":
                block_dict = metadata.get("block")
                ledger_name = metadata.get("ledger_name")
                if ledger_name not in ledgers:
                    logger.info("Invalid ledger")
                    client_sock.sendall(b"ERROR")
                    return

                block = Block.from_dict(block_dict)
                ledger = ledgers[ledger_name]

                try:
                    ledger.accept_block(block)
                except ValueError as exc:
                    logger.warning(f"Rejected replicated block: {exc}")
                    client_sock.sendall(b"ERROR")
                    return

                client_sock.sendall(b"OK")
                logger.info(f"Replicated block {block.index} into {ledger_name}")

            else:
                client_sock.sendall(b"ERROR")
                logger.error(f"Invalid metadata type from {addr}")

        except Exception as e:
            logger.error(f"Error receiving model from {addr}: {e}")


# TODO: use asyncio.create_task() for scalability, look into scalability

'''
TODO: need to save model hospital wise
'''
@app.post("/submit_local_model")
async def submit_local_model(req: Request):
    metadata = json.loads(req.headers.get("metadata"))
    node_id = metadata.get("node_id")
    model_id = metadata.get("model_id")

    logger.info(f"Model upload received from: {node_id}")

    node_queues = {}
    threads = []
    sockets = {}

    bc_nodes = list(bc_nodes_registry.items())
    results = {node: None for node, _ in bc_nodes}

    '''
    here, we are essentially creating a message queue for each node in the blockchain netowrk thats alive.
    for each node, a thread is opened which pulls chunks from the queue and sends to the node.

    this creates a mechanism which makes sure parallel, in order delivery to each node at its own pace.
    '''

    def send_wrapper(node, info, q):
        result = send_to_node(node, info, q)
        if result:
            results[node], sockets[node] = result
        else:
            results[node] = None

    for node, info in bc_nodes:
        q = queue.Queue()
        node_queues[node] = q

        t = threading.Thread(target=send_wrapper, args=(node, info, q), daemon=True)
        t.start()
        threads.append(t)

    hasher = hashlib.sha256()

    # leader keeps its own copy too, so the hospital can fetch it back for aggregation
    os.makedirs(f"{MODEL_STORE}/stage", exist_ok=True)
    os.makedirs(f"{MODEL_STORE}/commit", exist_ok=True)
    stage_path = f"{MODEL_STORE}/stage/leader_{time.time()}.pt"

    with open(stage_path, "wb") as f:
        async for chunk in req.stream():
            hasher.update(chunk)
            f.write(chunk)

            for q in node_queues.values():
                q.put(chunk)

    for q in node_queues.values():
        # basically an EOF flag
        q.put(None)

    for t in threads:
        t.join(timeout=30)

    model_hash = hasher.hexdigest()
    logger.info(f"Model hash computed: {model_hash}")

    logger.info(f"results: {results}")
    
    HM = set(h for h in results.values() if h is not None)
    # every peer must have returned the same hash as the leader. if a peer failed (None) or no
    # peer answered at all, nothing was verified, so it is not consensus
    all_same = len(HM) == 1 and model_hash in HM and None not in results.values()
    verified_count = sum(1 for h in results.values() if h == model_hash)
    total_count = len(results)
    
    logger.info(f"Consensus check: all_same={all_same}, {verified_count}/{total_count} verified")
    if HM:
        logger.info(f"Hash set HM: {HM}")
    
    # Get hospital_id from metadata (sent by h_node)
    hospital_id = metadata.get("hospital_id", "hospital_0")
    
    signal = b"COMMIT" if all_same else b"REJECT"
    for node, sock in sockets.items():
        try:
            sock.sendall(signal)
        except Exception as e:
            logger.error(f"Error in 2 phase commit: {e}")
        finally:
            sock.close()

    if all_same:
        os.replace(stage_path, f"{MODEL_STORE}/commit/{model_hash}.pt")
    else:
        os.remove(stage_path)

    if all_same:
        payload = {
            "hospital_id": hospital_id,
            "node_id": metadata.get("node_id"),
            "model_hash": model_hash,
            "model_id": model_id,
            "round": metadata.get("round"),
            "kind": metadata.get("kind", "local"),     # "local" = M_i from an edge server, "global" = GM_i
            "metrics": metadata.get("metrics", {}),
            "trust_score": 1.0
        }

        logger.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!CONSENSUS MADE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        ledger_name = hospital_id if hospital_id in ledgers else "hospital_0"
        logger.info(f"ledgers: {ledgers}, ledger_name: {ledger_name}")
        block = ledgers[ledger_name].append_block(payload)
        logger.info(f"Block {block.index} added to {ledger_name} blockchain")

        leader_name = os.getenv("BC_NODE_NAME")
        for node in bc_nodes_registry:
            if node == leader_name:
                continue
            threading.Thread(
                target=replicate_block,
                args=(node, ledger_name, block.to_dict()),
                daemon=True
            ).start()

        return {
            "status": "success",
            "model_hash": model_hash,
            "block_index": block.index,
            "blockchain": ledger_name,
            "consensus": {
                "all_same": all_same,
                "verified": verified_count,
                "total": total_count
            }
        }
    else:
        logger.error(f"Consensus failed! Hashes don't match.")
        return {
            "status": "consensus_failed",
            "model_hash": model_hash,
            "consensus": {
                "all_same": all_same,
                "verified": verified_count,
                "total": total_count,
                "hash_set": list(HM)
            }
        }

def replicate_block(node, ledger_name, block):
    try:
        info = bc_nodes_registry[node]
        sock = socket(AF_INET, SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((node, info["broadcast_port"]))
        msg = json.dumps({
            "type": "block_replication",
            "ledger_name": ledger_name,
            "block": block
        })
        sock.sendall(msg.encode())
        ack = sock.recv(1024).decode().strip()
        sock.close()
        if ack == "OK":
            logger.info(f"Replicated block to {node}")
        else:
            logger.error(f"Replication rejected by {node}: {ack}")
    except Exception as e:
        logger.error(f"Failed to replicate block to {node}: {e}")

# ---------------- read access for hospitals ----------------

@app.get("/ledger/{ledger_name}")
def get_ledger(ledger_name: str):
    if ledger_name not in ledgers:
        raise HTTPException(404, "Unknown ledger")
    logger.info(f"Serving {ledger_name} ledger ({len(ledgers[ledger_name].chain)} blocks)")
    return [b.to_dict() for b in ledgers[ledger_name].chain]


@app.get("/model/{model_hash}")
def get_committed_model(model_hash: str):
    # hash comes from the ledger; only allow hex so this can't be used to read other files
    if not re.fullmatch(r"[0-9a-f]{64}", model_hash):
        raise HTTPException(400, "Invalid model hash")
    path = f"{MODEL_STORE}/commit/{model_hash}.pt"
    if not os.path.exists(path):
        logger.error(f"Model {model_hash} requested but never committed")
        raise HTTPException(404, "Model not committed")
    logger.info(f"Serving committed model {model_hash} to hospital")
    return FileResponse(path, media_type="application/octet-stream")


# ---------------- multi-institutional blockchain (B_M) ----------------
#
# Each hospital sends the output probabilities of its global model on the shared
# val/test splits. Once all hospitals have submitted, every B_M node runs the grid
# search (Alg. 5) on the same inputs; if all result hashes agree, the weights and
# the set of global model ids are appended to the multi_inst ledger (Alg. 6, Fig. 5).
#
# NOTE: in the paper the probabilities come out of SMPC encrypted inference
# (Alg. 2-3). Here each hospital computes them in the clear.

ensemble_submissions = {}
ensemble_lock = threading.Lock()


@app.post("/compute_ensemble_weights")
def compute_ensemble_weights(submissions: dict):
    logger.info(f"Leader asked to tune ensemble weights for {sorted(submissions)}")
    result = tune_ensemble(submissions)
    h = result_hash(result)
    logger.info(f"My ensemble result: weights {result['weights']}, hash {h}, sending back to leader")
    return {"result": result, "hash": h}


@app.post("/submit_ensemble_probs")
def submit_ensemble_probs(submission: dict):
    hospital_id = submission["hospital_id"]

    with ensemble_lock:
        ensemble_submissions[hospital_id] = submission
        logger.info(f"Ensemble probs received from {hospital_id} ({len(ensemble_submissions)}/{NUM_HOSPITALS})")
        if len(ensemble_submissions) < NUM_HOSPITALS:
            return {"status": "waiting", "received": sorted(ensemble_submissions)}
        submissions = dict(ensemble_submissions)
        ensemble_submissions.clear()

    logger.info("!!!!!!!!!!ALL HOSPITALS SUBMITTED, STARTING ENSEMBLE WEIGHT TUNING!!!!!!!!!!")

    # every B_M node tunes the weights independently
    result = tune_ensemble(submissions)
    hashes = {os.getenv("BC_NODE_NAME"): result_hash(result)}
    logger.info(f"Leader's ensemble result: weights {result['weights']}, hash {hashes[os.getenv('BC_NODE_NAME')]}")
    for node, info in bc_nodes_registry.items():
        logger.info(f"Sending probabilities to {node} so it can tune the weights on its own")
        try:
            res = httpx.post(f"{info['backend_url']}/compute_ensemble_weights", json=submissions, timeout=300)
            res.raise_for_status()
            hashes[node] = res.json()["hash"]
            logger.info(f"Received ensemble hash from {node}: {hashes[node]}")
        except Exception as e:
            logger.error(f"Ensemble tuning failed on {node}: {e}")
            hashes[node] = None

    HM = set(hashes.values())
    logger.info(f"Ensemble consensus hashes: {hashes}")
    if len(HM) != 1 or None in HM:
        logger.error("Ensemble consensus failed! Hashes don't match.")
        return {"status": "consensus_failed", "hashes": hashes}

    block = ledgers["multi_inst"].append_block(result)
    logger.info(f"!!!!!!!!!!ENSEMBLE CONSENSUS MADE!!!!!!!!!! Block {block.index} added to multi_inst: {result}")

    for node in bc_nodes_registry:
        threading.Thread(
            target=replicate_block,
            args=(node, "multi_inst", block.to_dict()),
            daemon=True
        ).start()

    logger.info(format_summary(result, block.index, block.block_hash, f"{len(hashes)}/{len(hashes)}"))

    return {"status": "success", "block_index": block.index, "result": result}


@app.post("/record_trust")
def record_trust(report: dict):
    # twin trust report of one round, kept on the hospital's own ledger so it can be audited later
    ledger_name = report["hospital_id"]
    block = ledgers[ledger_name].append_block({"type": "TRUST", "kind": "trust", "round": report["round"], "nodes": report["nodes"]})
    logger.info(f"Trust report for round {report['round']} added to {ledger_name} as block {block.index}")

    for node in bc_nodes_registry:
        threading.Thread(target=replicate_block, args=(node, ledger_name, block.to_dict()), daemon=True).start()

    return {"status": "success", "block_index": block.index}


@app.get("/ensemble_result")
def get_ensemble_result():
    chain = ledgers["multi_inst"].chain
    if len(chain) < 2:
        raise HTTPException(404, "No ensemble weights on the multi-institutional blockchain yet")
    return chain[-1].to_dict()
