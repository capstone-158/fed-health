import os
import json
import time
import threading
import logging
from backend import checkin, register_nodes, recv_model_and_hash
from shared.commons import *
from shared.config import BLOCKCHAIN_PORTS
import uvicorn

rounds_completed = 0

logger = get_logger()
logger.info("Starting blockchain node server...")

leader = os.getenv("LEADER")
logger.info(f"LEADER value being returned is {leader} and type is {type(leader)}")
'''
registry should look like:
{
    bc_node_1 : {
        "broadcast_port": 7000,         # suppose
        "backend_url": http://bc_node_1:5000
    },

    bc_node_2 : {
        "broadcast_port": 7000,         # suppose
        "backend_url": http://bc_node_2:5100
    }
}
'''

# TODO: add a thread to listen for broadcasts of models

if leader == "true":
    registry_thread = threading.Thread(target=register_nodes, daemon=True)
    registry_thread.start()
    logger.info("Started thread to listen for checkins")
else:
    checkin()

broadcast_thread = threading.Thread(target=recv_model_and_hash, daemon=True)
broadcast_thread.start()

logger.info(f"Starting FastAPI server on port {BLOCKCHAIN_PORTS['backend_port']}")
while True:
    try:
        uvicorn.run(
            "backend:app",
            host="0.0.0.0",
            port=BLOCKCHAIN_PORTS["backend_port"],
            reload=False
        )
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        logger.info("Restarting in 5 seconds...")
        time.sleep(5)
