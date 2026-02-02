# this is where all the threads run
import time
import threading
import logging
from main import heartbeat_loop

rounds_completed = 0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.info("Starting edge Server...")
status_thread = threading.Thread(target=heartbeat_loop, daemon=True)
status_thread.start()

while True:
    time.sleep(60)