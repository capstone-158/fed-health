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

from fastapi import FastAPI, Request, HTTPException
import json
import hashlib
import os
from socket import *
from shared.config import BLOCKCHAIN_PORTS, BLOCKCHAIN_LEADER_HOSTNAME, MODEL_CHUNK_SIZE
from shared.commons import *
import time
from ping3 import ping
import threading

logger = get_logger()
app = FastAPI()

os.makedirs("/models")

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
        sock.settimeout(5) 
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

def send_to_node(node, info, model_path, local_hash):
    try:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((node, info["broadcast_port"]))
        
        metadata = {
            "type": "model_broadcast",
            "node": node,
            "info": info
        }
        sock.sendall(json.dumps(metadata).encode())

        ack = sock.recv(1024)
        
        if ack != b"OK":
            logger.error(f"Node {node} not ready, received: {ack}")
            sock.close()
            return None

        with open(model_path, "rb") as f:
            while True:
                chunk = f.read(MODEL_CHUNK_SIZE)
                if not chunk:
                    break
                sock.sendall(chunk)
        
        sock.sendall(b"EOF")

        response_hash = sock.recv(1024).decode()
        logger.info(f"Received hash from {node}: {response_hash}")

        sock.close()
        
        if response_hash == local_hash:
            logger.info(f"Hash verified for {node}")
        else:
            logger.error(f"Hash mismatch for {node}! Expected {local_hash}, got {response_hash}")
        
        return response_hash
            
    except Exception as e:
        logger.error(f"Failed to broadcast to {node}: {e}")
        return None

# TODO: somehow see if it can be done parallely
def broadcast(model_path, local_hash):
    threads = []
    bc_nodes = bc_nodes_registry.items()
    results = {}

    def send_wrapper(node, info):
        results[node] = send_to_node(node, info, model_path, local_hash)

    for node, info in bc_nodes:
        if ping(node) is not None:
            t = threading.Thread(target=send_wrapper, args=(node, info))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()
    
    verified_count = sum(1 for h in results.values() if h == local_hash)
    total_count = len(results)
    
    logger.info(f"Broadcast complete: {verified_count}/{total_count} nodes verified")
    
    return verified_count, total_count, results

def recv_model_and_hash():
    try:
        sock = create_socket("0.0.0.0", BLOCKCHAIN_PORTS["broadcast_port"])
        sock.listen()
        
        logger.info(f"Listening for model broadcasts on port {BLOCKCHAIN_PORTS['broadcast_port']}")
        
        while True:
            client_sock, addr = sock.accept()
            
            with client_sock:
                try:
                    metadata_raw = client_sock.recv(4096).decode()
                    metadata = json.loads(metadata_raw)
                    
                    logger.info(f"Received broadcast metadata from {addr}: {metadata}")
                    
                    if metadata.get("type") == "model_broadcast":
                        client_sock.sendall(b"OK")
                        logger.info(f"Sent OK to {addr}, ready to receive model")
                    else:
                        client_sock.sendall(b"ERROR")
                        logger.error(f"Invalid metadata type from {addr}")
                        continue
                    
                    hasher = hashlib.sha256()
                    
                    while True:
                        chunk = client_sock.recv(MODEL_CHUNK_SIZE)
                        if not chunk:
                            break
                        
                        if chunk == b"EOF":
                            break
                        
                        hasher.update(chunk)
                    
                    model_hash = hasher.hexdigest()
                    logger.info(f"Received model from {addr}, hash: {model_hash}")
                    
                    client_sock.sendall(model_hash.encode())
                    
                except Exception as e:
                    logger.error(f"Error receiving model from {addr}: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start model receiver: {e}")



# TODO: fix this api later
@app.post("/submit_local_model")
async def submit_local_model(req: Request):
    # TODO: DONT STORE THE MODEL, FORWARD CHUNKS AS THEY COME
    metadata = json.loads(req.headers.get("metadata"))
    node_id = metadata.get("node_id")
    model_id = metadata.get("model_id")

    logger.info(f"Model upload received from: {node_id}")

    hasher = hashlib.sha256()
    
    with open(f"/models/{model_id}", "ab") as f:
        async for chunk in req.stream():
            hasher.update(chunk)
            f.write(chunk)
        model_hash = hasher.hexdigest()

    os.rename(f"/models/{model_id}", f"/models/{model_hash}")