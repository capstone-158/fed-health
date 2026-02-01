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

logger.info("Starting edge Server...")

while True:
    time.sleep(60)