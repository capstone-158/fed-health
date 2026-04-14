# import torch
import os
ROUNDS=10
EPOCHS=20
LEARNING_RATE=0.001
NUM_CLASSES=9
CLIENTS=2   #clients per round
BATCH_SIZE=128
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HOSPITAL_BACKEND_PORT=8000
HOSPITAL_URL=f"http://{os.getenv('H_HOST')}:{HOSPITAL_BACKEND_PORT}"
HOSPITAL_HEARTBEAT_PORT=4000

HEARTBEAT_INTERVAL=1

MODEL_CHUNK_SIZE=8192*1024*1024

BLOCKCHAIN_LEADER_HOSTNAME="bc_node_0"
BLOCKCHAIN_PORTS = {
    "backend_port": 5000,
    "leader_checkin_port": 8000,
    "broadcast_port": 7000
}
BLOCKCHAIN_BACKEND_URL=f"http://{BLOCKCHAIN_LEADER_HOSTNAME}:{BLOCKCHAIN_PORTS.get('backend_port')}"

NODE_REGISTRY = {
    "0": "hospital_0",
    "1": "hospital_0",
    "2": "hospital_1",
}
