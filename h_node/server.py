# this is where all the threads run
import time
import threading
import logging
from heartbeat_handler import listen

rounds_completed = 0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.info("Starting hospital server...")

heartbeat_thread = threading.Thread(target=listen, daemon=True)
heartbeat_thread.start()

while True:
    time.sleep(60)