import redis
import json
import os
from shared.config import HEARTBEAT_INTERVAL
from shared.commons import *

logger = get_logger()

def get_connection():
    redis_host = os.getenv("REDIS_HOST")
    r = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
    return r

r = get_connection()

def insert_status(hospital_id, node_id, payload):
    '''
    Assuming payload looks like:
        payload = {
            "state": "training",      # idle | training | uploading | error
            "round": "2",
            "epoch": "11"
        }
    '''

    key = f"hospital_id:{hospital_id}:node:{node_id}"
    payload_serialised = {
        'node_id': str(payload.get('node_id')),
        'state': payload.get('state'),
        'timestamp': str(payload.get('timestamp')),
        'payload': json.dumps(payload.get('payload'))
    }
    try:
        r.hset(key, mapping=payload_serialised)
        r.expire(key, 2*HEARTBEAT_INTERVAL)
    except Exception as e:
        logger.error(e)
