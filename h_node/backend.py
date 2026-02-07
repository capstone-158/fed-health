'''
systems stuff:
- write to blockchain using gRPCs

api:
- POST/check_in()
- POST/upload_model()
- POST/send_global_model()

fed apis():
- GET/get_init_weights
- 
'''

'''
Workflow:
- send random init weights from central server to nodes
- pick any K nodes for participation.
- train model for E epochs ---------------------- update blockchain
- send to central server. it aggregates --------- update blockchain
- replace the global model
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

@app.post("/upload_model")
async def upload_model(req: Request):
    metadata_str = req.headers.get("metadata")
    if not metadata_str:
        raise HTTPException(status_code=400, detail="Missing metadata header")
    
    metadata = json.loads(metadata_str)
    
    # Add hospital ID to metadata
    hospital_id = os.getenv("H_NODE_NAME", "hospital_0")
    metadata["hospital_id"] = hospital_id
    
    node_id = metadata.get("node_id")
    model_id = metadata.get("model_id")
    
    logger.info(f"Received model from node {node_id}, model_id: {model_id}")
    logger.info(f"Forwarding to blockchain leader: {BLOCKCHAIN_LEADER_HOSTNAME}")
    
    bc_leader_url = f"http://{BLOCKCHAIN_LEADER_HOSTNAME}:{BLOCKCHAIN_PORTS['backend_port']}/submit_local_model"
    
    async def stream_chunks():
        async for chunk in req.stream():
            yield chunk
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                bc_leader_url,
                headers={"metadata": json.dumps(metadata)},  # Use updated metadata with hospital_id
                content=stream_chunks()
            )
            
            bc_response = response.json()
            
            logger.info(f"Blockchain response: {bc_response}")
            
            return {
                "status": "success",
                "message": "Model uploaded to blockchain",
                "blockchain_response": bc_response
            }
            
    except Exception as e:
        logger.error(f"Failed to forward model to blockchain: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to forward to blockchain: {str(e)}"
        )
