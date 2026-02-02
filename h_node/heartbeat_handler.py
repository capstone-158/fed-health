from datetime import datetime
import os
import json
import time
import threading
from socket import *
import logging
from shared.commons import create_socket
from shared.config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

node_registry = {}

def listen():
    logger = logging.getLogger(__name__)
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

            if data:
                status = json.loads(data.decode())
                node_id = status.get("node_id")

                logger.info(f"Got heartbeat from {client_addr}")
                
                # Store in registry (Digital Twin)
                node_registry[node_id] = status
                logger.info(f"Heartbeat from node_{node_id}: {status['status']}")

    except Exception as e:
        logger.error(f"Heartbeat listener error: {e}")