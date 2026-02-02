from datetime import datetime
import os
import json
import time
import threading
from socket import *
import logging
import redis
from shared.commons import *
from shared.config import *
import db_utils

logger = get_logger()

def listen():
    h_id = os.getenv("H_ID", "unknown")
    logger.info(f"Starting heartbeat listener on h_node_{h_id}")

    try:
        sock = create_socket("0.0.0.0", HOSPITAL_HEARTBEAT_PORT)
        sock.listen(5)
        logger.info(f"Hospital {h_id} ready to receive heartbeats on port {HOSPITAL_HEARTBEAT_PORT}...")

        while True:
            client_sock, client_addr = sock.accept()
            
            data = client_sock.recv(1024)
            client_sock.close()

            '''
                data should have hospital_id, node_id, payload.
                data = {
                    hospital_id: 0,
                    node_id: 2,
                    status: training
                    timestamp: datetime.datetime.now()
                    payload: {
                        epoch: 13,
                        round: 2,
                        last_seen: time.time()
                    }
                }
            '''

            if data:
                status = json.loads(data.decode())
                node_id = status.get("node_id")

                logger.info(f"Got heartbeat from {client_addr}")
                logger.info(f"payload written is: {status}")
                db_utils.insert_status(h_id, node_id, status)
                logger.info(f"Heartbeat from node_{node_id}: {status.get('state')}")

    except Exception as e:
        logger.error(f"Heartbeat listener error: {e}")