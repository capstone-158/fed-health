from datetime import datetime
import os
import json
import time
import threading
from socket import *
import logging
from shared.commons import *
from shared.config import *
from shared import state_manager

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
                # node puts round/epoch inside "payload", not at the top level
                payload = status.get("payload", {})

                logger.info(f"Got heartbeat from {client_addr}")
                logger.info(f"payload written is: {status}")
                state_manager.update_node_status(
                    h_id, node_id, 
                    state=status.get("state"),
                    round_id=payload.get("round"),
                    epoch=payload.get("epoch")
                )
                if status.get("state") == "training" and payload.get("round") is not None:
                    state_manager.record_twin_training(h_id, node_id, payload["round"], payload.get("epoch"))
                logger.info(f"Heartbeat from node_{node_id}: {status.get('state')}")

    except Exception as e:
        logger.error(f"Heartbeat listener error: {e}")