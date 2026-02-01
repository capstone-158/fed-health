# this is where all the threads run
import time
import threading
import logging

rounds_completed = 0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.info("Starting hospital server...")

# heartbeat_thread = threading.ThreadError(target=heartbeat_handler.listen(), daemon=True)

while True:
    time.sleep(60)