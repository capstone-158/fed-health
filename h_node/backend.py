'''
systems stuff:
- Aggregation of local models received from blockchain
- Distribute global model back to edge servers

api:
- POST/receive_aggregated_model() - receive model from blockchain for aggregation
- GET/get_global_model() - edge servers retrieve aggregated global model

Fed FL Workflow:
- edge nodes train local models
- edge nodes upload DIRECTLY to blockchain
- blockchain verifies and stores model hashes
- central server (hospital) retrieves models from blockchain
- central server aggregates models (FedAvg, etc.)
- central server distributes global model back to edge servers
REPEAT FOR T ROUNDS
'''

from fastapi import FastAPI, Request, HTTPException
import httpx
import json
import os
from shared.config import BLOCKCHAIN_LEADER_HOSTNAME, BLOCKCHAIN_PORTS
from shared.commons import get_logger

logger = get_logger()
app = FastAPI()

# TODO: Implement model aggregation logic (FedAvg)
# TODO: Implement global model distribution to edge servers
