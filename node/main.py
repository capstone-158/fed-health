from socket import *
import os
import json
import time
import threading
from datetime import datetime
from shared.config import HOSPITAL_HEARTBEAT_PORT, HEARTBEAT_INTERVAL
from shared.commons import create_socket

def send_status():
    host_hospital = os.getenv("H_HOST")
    node_id = os.getenv("NODE_ID", "unknown")
    
    try:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect((host_hospital, HOSPITAL_HEARTBEAT_PORT))

        status = {
            "node_id": node_id,
            "timestamp": datetime.now().isoformat(),
            "status": "idle"
        }
        
        sock.send(json.dumps(status).encode())
        sock.close()
        
    except Exception as e:
        print(f"Failed to send heartbeat: {e}")

def heartbeat_loop():
    while True:
        send_status()
        time.sleep(HEARTBEAT_INTERVAL)