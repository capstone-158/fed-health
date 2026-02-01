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

def listen():
    logger = logging.getLogger(__name__)
    logger.info(f"Starting heartbeat listener on {os.getenv("H_ID")}")

    try:
        sock = create_socket("0.0.0.0", HOSPITAL_HEARTBEAT_PORT)
        sock.listen(5)
        logger.info(f"Hospital {os.getenv("H_ID")} ready to receive heartbeats...")

        while True:
            client_sock, client_addr = sock.accept()
            logger.debug(f"Got connection from {client_addr}")

            data = client_sock.recv(1024)

            if data:
                json_data = json.loads(data.decode())
                print("Data: ",data)

    except Exception as e:
        logger.error(e)

