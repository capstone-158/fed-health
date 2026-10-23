# import torch
import os
# set per data mode by envs/*.env through docker-compose
ROUNDS=int(os.getenv("ROUNDS", 10))
EPOCHS=int(os.getenv("EPOCHS", 20))
LEARNING_RATE=0.001
# same optimizer as the centralised baselines in the training notebooks (SGD, not Adam).
# Adam at 0.001 left pretrained AlexNet stuck predicting one class.
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
# 9 for colon pathology (PathMNIST), 2 for breast tumour (BreakHis); set per mode in envs/*.env
NUM_CLASSES=int(os.getenv("NUM_CLASSES", 9))
CLIENTS=2   #clients per round
BATCH_SIZE=int(os.getenv("BATCH_SIZE", 128))     # effective batch size per optimizer step (paper: 128)
# batch that actually goes through the model. if smaller than BATCH_SIZE, gradients are
# accumulated over BATCH_SIZE // MICRO_BATCH steps -> same update as BATCH_SIZE, less memory
MICRO_BATCH=int(os.getenv("MICRO_BATCH", BATCH_SIZE))
# how long a hospital waits for its nodes to finish one round before marking it failed
ROUND_TIMEOUT=int(os.getenv("ROUND_TIMEOUT", 600))
NUM_HOSPITALS=3

# shared evaluation splits every hospital runs its global model on for ensemble tuning.
# /app/data is whichever data folder the env file mounts (data/tiny, data/small or data/)
VAL_DATA_PATH="/app/data/val.pt"
TEST_DATA_PATH="/app/data/test.pt"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defaults are for docker; run_native.py overrides them for nodes running natively on macOS
HOSPITAL_BACKEND_PORT=8000
HOSPITAL_URL=os.getenv("HOSPITAL_URL", f"http://{os.getenv('H_HOST')}:{HOSPITAL_BACKEND_PORT}")
HOSPITAL_HEARTBEAT_PORT=int(os.getenv("HOSPITAL_HEARTBEAT_PORT", 4000))

HEARTBEAT_INTERVAL=10

MODEL_CHUNK_SIZE=8192*1024*1024

BLOCKCHAIN_LEADER_HOSTNAME="bc_node_0"
BLOCKCHAIN_PORTS = {
    "backend_port": 5000,
    "leader_checkin_port": 8000,
    "broadcast_port": 7000
}
BLOCKCHAIN_BACKEND_URL=os.getenv("BLOCKCHAIN_URL", f"http://{BLOCKCHAIN_LEADER_HOSTNAME}:{BLOCKCHAIN_PORTS.get('backend_port')}")

NODE_REGISTRY = {
    "0": "hospital_0",
    "1": "hospital_0",
    "2": "hospital_1",
}
