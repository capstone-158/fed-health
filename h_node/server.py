# this is where all the threads run
import time
import uvicorn
import threading
import logging
from heartbeat_handler import listen
from crypto import *    # custom module
from shared.commons import *
import os
from cryptography.hazmat.primitives import serialization

logger = get_logger()

os.makedirs("./secrets", exist_ok=True)

private_key_path = "./secrets/private_key.pem"
public_key_path = "./secrets/public_key.pem"

if os.path.exists(private_key_path) and os.path.exists(public_key_path):
    with open(private_key_path, "rb") as f:
        priv_pem = f.read()
    with open(public_key_path, "rb") as f:
        pub_pem = f.read()
    
    priv_key = serialization.load_pem_private_key(priv_pem, password=None)
else:
    priv_key, priv_pem, pub_pem = generate_keys()
    
    with open(private_key_path, "wb") as f:
        f.write(priv_pem)
    
    with open(public_key_path, "wb") as f:
        f.write(pub_pem)
    
rounds_completed = 0

import backend
backend.priv_key = priv_key
backend.pub_pem = pub_pem

logger.info("Starting hospital server...")

heartbeat_thread = threading.Thread(target=listen, daemon=True)
heartbeat_thread.start()

while True:
    try:
        uvicorn.run(
            "backend:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        logger.info("Restarting in 5 seconds...")
        time.sleep(5)
