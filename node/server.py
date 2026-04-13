import os

# this is where all the threads run
import time
import threading
import json
from main import heartbeat_loop
from shared.commons import *
import os
from shared import state_manager
from main import handle_training, node_state
import asyncio
from training.ml_utils import client_update

logger = get_logger()
logger.info(type(os.getenv("NODE_ID")))

rounds_completed = 0

logger.info("Starting edge Server...")
status_thread = threading.Thread(target=heartbeat_loop, daemon=True)
status_thread.start()

# TODO: add a while look to wait for a main signal. upon recieving the main signal, only then the real training stuff starts
def listen(channel_name, redis_obj):
    pubsub = redis_obj.pubsub()
    pubsub.subscribe(channel_name)
    logger.info(f"Node listening on Redis channel: {channel_name}")

    for msg in pubsub.listen():
        if msg['type'] == 'message':
            try:
                command = json.loads(msg['data'])
                logger.info(f"Received command: {command}")
                
                if command.get('type') == 'start training':
                    node_id = os.getenv('NODE_ID')
                    logger.info(f"im node id: {node_id} with type: {type(node_id)} and am i in {command.get('node_ids')}?")
                    if node_id in command.get('node_ids', []):
                        logger.info(f"Node {node_id} SELECTED for round {command.get('round_id')}")
                        logger.info(f"Starting training with epochs={command.get('epochs')}, model_ref={command.get('model_ref')}")

                        '''
                            heres where the actual training logic and stuff happens.
                            call a function called handle_trainig() or smtg to do all the stuff.
                            - loads dataset.
                            - load the global model.
                            - starts training model
                                - change state to "training" and keep sending heartbeats
                                - keep training and sending
                            - evaluate metrics
                            - uploads model with metrics to the blockchain
                            - calls the barrier_checkin() function saying that its done.
                        '''
                        asyncio.run(handle_training(epoch_num=command.get("epochs"), round_num=command.get("round_id"), node_state=node_state))

                    else:
                        logger.info(f"Node {node_id} not selected for this round")
                else:
                    logger.debug(f"Received command type: {command.get('type')}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}, raw data: {msg['data']}")
        else:
            logger.debug(f"Received non-message event: {msg['type']}")

red = state_manager.get_connection()
sub_thread = threading.Thread(target=listen, daemon=True, args=("fl_commands", red))
sub_thread.start()

while True:
    time.sleep(60)