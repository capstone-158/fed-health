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
import queue
from socket import *
from shared.config import BLOCKCHAIN_PORTS, BLOCKCHAIN_LEADER_HOSTNAME, MODEL_CHUNK_SIZE
from shared.commons import *
import time
from ping3 import ping
import threading
from ledger import ledgers, Block
import httpx

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
        sock.connect((node, info["broadcast_port"]))  # Must be a tuple

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
            if chunk == None:
                break
            sock.sendall(chunk)

        sock.sendall(b"EOF")

        response_hash = sock.recv(1024).decode()
        logger.info(f"Received hash from {node}: {response_hash}")
        
        sock.close()

        return response_hash

    except Exception as e:
        logger.error(f"Error in sending chunk to node. Error: {e}")
        return None



# THIS FUNCTION IS FOR THE RECEIVER SIDE
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
                        
                        if chunk.endswith(b"EOF"):
                            if len(chunk) > 3:
                                hasher.update(chunk[:-3])
                            break
                        
                        hasher.update(chunk)
                    
                    model_hash = hasher.hexdigest()
                    logger.info(f"Received model from {addr}, hash: {model_hash}")
                    
                    client_sock.sendall(model_hash.encode())
                    
                except Exception as e:
                    logger.error(f"Error receiving model from {addr}: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start model receiver: {e}")


# TODO: use asyncio.create_task() for scalability, look into scalability
@app.post("/submit_local_model")
async def submit_local_model(req: Request):
    # TODO: DONT STORE THE MODEL, FORWARD CHUNKS AS THEY COME
    metadata = json.loads(req.headers.get("metadata"))
    node_id = metadata.get("node_id")
    model_id = metadata.get("model_id")

    logger.info(f"Model upload received from: {node_id}")


    node_queues = {}
    threads = []
    results = {}

    bc_nodes = list(bc_nodes_registry.items())

    '''
    here, we are essentially creating a message queue for each node in the blockchain netowrk thats alive.
    for each node, a thread is opened which pulls chunks from the queue and sends to the node.

    this creates a mechanism which makes sure parallel, in order delivery to each node at its own pace.
    '''
    
    def send_wrapper(node, info, q):
        """Wrapper to capture return value from send_to_node"""
        results[node] = send_to_node(node, info, q)
    
    for node, info in bc_nodes:
        q = queue.Queue()
        node_queues[node] = q

        t = threading.Thread(target=send_wrapper, args=(node, info, q))
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
        t.join(timeout=30)  # 30 second timeout per thread

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
        
        # Append to blockchain
        ledger_name = hospital_id if hospital_id in ledgers else "hospital_0"
        logger.info(f"ledgers: {ledgers}, ledger_name: {ledger_name}")
        block = ledgers[ledger_name].append_block(payload)
        logger.info(f"✓ Block {block.index} added to {ledger_name} blockchain")

        leader_name = os.getenv("BC_NODE_NAME")
        for node, info in bc_nodes_registry.items():
            if node == leader_name:
                continue

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"http://{node}:{BLOCKCHAIN_PORTS['backend_port']}/replicate_block",
                        json={
                            "ledger_name": ledger_name,
                            "block": block.to_dict()
                        }
                    )
                logger.info(f"Replicated block to {node}")
            except Exception as e:
                logger.error(f"Failed to replicate block to {node}: {e}")

        
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
        logger.error(f"✗ Consensus failed! Hashes don't match.")
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
    
# TODO: CHANGE TO SOCKETS. DELETE THIS API
@app.post("/replicate_block")
async def replicate_block(data: dict):
    ledger_name = data.get("ledger_name")
    block_dict = data.get("block")
    if ledger_name not in ledgers:
        raise HTTPException(status_code=400, detail="Invalid ledger")

    block = Block.from_dict(block_dict)
    ledger = ledgers[ledger_name]
    last_block = ledger.last_block()
    
    if block.prev_hash != last_block.block_hash:
        raise HTTPException(status_code=400, detail="Chain mismatch")
    with ledger.lock:
        ledger.chain.append(block)
        ledger._persist()

    logger.info(f"Replicated block {block.index} into {ledger_name}")
    return {"status": "replicated"}