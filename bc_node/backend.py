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
from fastapi import FastAPI, Request
import json
import hashlib
import os
import queue
from socket import *
from shared.config import BLOCKCHAIN_PORTS, BLOCKCHAIN_LEADER_HOSTNAME, MODEL_CHUNK_SIZE
from shared.commons import *
import time
from ping3 import ping
import threading
from ledger import ledgers, Block

logger = get_logger()
app = FastAPI()

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

        os.makedirs("/app/client_models/stage", exist_ok=True)
        os.makedirs("/app/client_models/commit", exist_ok=True)

        while True:
            client_sock, addr = sock.accept()
            
            with client_sock:
                try:
                    metadata_raw = client_sock.recv(4096).decode()
                    metadata = json.loads(metadata_raw)
                    
                    logger.info(f"Received metadata from {addr}: {metadata}")
                    
                    if metadata.get("type") == "model_broadcast":
                        client_sock.sendall(b"OK")
                        logger.info(f"sent ok to {addr}, ready to receive model")

                        hasher = hashlib.sha256()
                        stage_path = f"/app/client_models/stage/model_{time.time()}.pt"

                        with open(stage_path, "wb") as f:
                            while True:
                                chunk = client_sock.recv(MODEL_CHUNK_SIZE)
                                if not chunk:
                                    break
                                
                                if chunk.endswith(b"eof"):
                                    if len(chunk) > 3:
                                        hasher.update(chunk[:-3])
                                    break
                                
                                hasher.update(chunk)
                                f.write(chunk)
                            
                            model_hash = hasher.hexdigest()
                            logger.info(f"received model from {addr}, hash: {model_hash}")
                            logger.info(f"Model: {model_hash} STAGED")

                            client_sock.sendall(model_hash.encode())
                        
                        final_path = f"/app/models/commit/{model_hash}.pt"
                        # listen for commit signal then save the model to a final path
                        signal = client_sock.recv(16).decode()
                        if signal == "COMMIT":
                            shutil.move(stage_path, final_path)
                        else:
                            os.remove(stage_path)
                        
                        logger.info(f"Model: {model_hash} COMMITTED")
                    
                    elif metadata.get("type") == "block_replication":
                        block_dict = metadata.get("block")
                        ledger_name = metadata.get("ledger_name")
                        if ledger_name not in ledgers:
                            logger.info("Invalid ledger")
                            client_sock.sendall(b"ERROR")
                            continue

                        block = Block.from_dict(block_dict)
                        ledger = ledgers[ledger_name]
                        last_block = ledger.last_block()

                        if block.prev_hash != last_block.block_hash:
                            logger.info("Chain mismatch")
                            client_sock.sendall(b"ERROR")
                            continue

                        with ledger.lock:
                            ledger.chain.append(block)
                            ledger._persist()

                        client_sock.sendall(b"OK")
                        logger.info(f"Replicated block {block.index} into {ledger_name}")

                    else:
                        client_sock.sendall(b"ERROR")
                        logger.error(f"Invalid metadata type from {addr}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error receiving model from {addr}: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start model receiver: {e}")


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
    results = {}
    sockets = {}

    bc_nodes = list(bc_nodes_registry.items())

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

    async for chunk in req.stream():
        hasher.update(chunk)

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
    all_same = len(HM) == 1 and model_hash in HM if HM else True 
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
        payload = {
            "hospital_id": hospital_id,
            "node_id": metadata.get("node_id"),
            "model_hash": model_hash,
            "model_id": model_id,
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