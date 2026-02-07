# this is where all the threads run
import time
import uvicorn
import threading
import logging
from heartbeat_handler import listen
from shared.commons import *

rounds_completed = 0
logger = get_logger()

logger.info("Starting hospital server...")

heartbeat_thread = threading.Thread(target=listen, daemon=True)
heartbeat_thread.start()

while True:
    time.sleep(60)
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
