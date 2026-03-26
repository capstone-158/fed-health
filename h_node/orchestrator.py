import os
import time
import random
import json
import logging
from shared.config import ROUNDS, EPOCHS, CLIENTS
from shared.state_manager import get_connection, set_current_round, set_round_state, wait_for_barrier, get_live_nodes
from shared.commons import get_logger

logger = get_logger()

def select_k_nodes():
    hospital_id = os.getenv("H_ID")
    live_nodes = get_live_nodes(hospital_id)

    if len(live_nodes) < CLIENTS:
        return live_nodes

    return random.sample(live_nodes, CLIENTS)

# def publish_start_round(round_id, node_ids, )