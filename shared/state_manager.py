import redis
import os
import json
import time
from shared.config import CLIENTS, HEARTBEAT_INTERVAL

def get_connection():
    r = redis.Redis(host=os.getenv("REDIS_HOST"), port=int(os.getenv("REDIS_PORT", 6379)), db=0, decode_responses=True)
    return r

r = get_connection()

# global round state
def set_current_round(round_id):
    r.set("fl:current_round", round_id)

# round state
def get_current_round():
    val = r.get("fl:current_round")
    return int(val) if val is not None else 0

def set_round_state(round_id, state):
    r.hset(f"fl:round:{round_id}", mapping={
        "state": state,
        "updated_at": time.time()
    })

def get_round_state(round_id):
    return r.hgetall(f"fl:round:{round_id}")

# -------------------Barrier Stuff---------------------

def barrier_checkin(round_id, node_id):
    # this is called by node when its done uploading
    r.sadd(f"barrier:round:{round_id}:done", node_id)

def barrier_count(round_id):
    return r.scard(f"barrier:round:{round_id}:done")

def wait_for_barrier(round_id, required=CLIENTS, poll_interval=5, timeout=600):
    # checks and blocks until all the K nodes have finished
    deadline = time.time() + timeout
    while time.time() < deadline:
        count = barrier_count(round_id)
        if count >= required:
            return True
        time.sleep(poll_interval)
    return False

# -------------------Hospital Stuff---------------------

def set_hospital_current_round(hospital_id, round_id):
    key = f"fl:hospital:{hospital_id}:current_round"
    r.set(key, round_id)

def get_hospital_current_round(hospital_id):
    key = f"fl:hospital:{hospital_id}:current_round"
    return int(r.get(key) or 0)

# changing node state
def update_node_status(hospital_id, node_id, state, round_id=None, epoch=None):
    key = f"fl:hospital:{hospital_id}:node:{node_id}"
    mapping = {
        "state": state,
        "timestamp": str(time.time())
    }

    if round_id is not None:
        mapping["round"] = str(round_id)
    if epoch is not None:
        mapping["epoch"] = str(epoch)

    r.hset(key, mapping=mapping)
    r.expire(key, 2 * HEARTBEAT_INTERVAL)

# -------------------Digital twin---------------------
# per node per round record, written by the hospital from what it sees. unlike the liveness key it doesnt expire

def record_twin_training(hospital_id, node_id, round_id, epoch):
    key = f"twin:hospital:{hospital_id}:node:{node_id}:round:{round_id}"
    now = time.time()
    r.hsetnx(key, "first_seen", now)
    r.hset(key, "last_seen", now)
    if epoch is not None:
        r.sadd(f"{key}:epochs", epoch)

def get_twin_round(hospital_id, node_id, round_id):
    key = f"twin:hospital:{hospital_id}:node:{node_id}:round:{round_id}"
    record = r.hgetall(key)
    epochs = r.smembers(f"{key}:epochs")
    first, last = float(record.get("first_seen", 0)), float(record.get("last_seen", 0))
    return {"epochs_seen": len(epochs), "train_seconds": last - first}

def get_trust(hospital_id, node_id):
    return float(r.hget(f"twin:hospital:{hospital_id}:node:{node_id}:profile", "trust") or 1.0)

def set_trust(hospital_id, node_id, trust):
    r.hset(f"twin:hospital:{hospital_id}:node:{node_id}:profile", "trust", trust)

def get_live_nodes(hospital_id):
    """Returns node_ids that have sent a heartbeat recently."""
    pattern = f"fl:hospital:{hospital_id}:node:*"
    keys = r.scan_iter(pattern)
    return [k.split(":")[-1] for k in keys]

