# this is where all the threads run
import time
import threading
from main import heartbeat_loop
from shared.commons import *

logger = get_logger()

rounds_completed = 0

logger.info("Starting edge Server...")
status_thread = threading.Thread(target=heartbeat_loop, daemon=True)
status_thread.start()

while True:
    time.sleep(60)