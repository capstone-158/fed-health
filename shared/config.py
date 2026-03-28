ROUNDS=10
EPOCHS=20
LEARNING_RATE=0.001
CLIENTS=2   #clients per round
BATCH_SIZE=128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HOSPITAL_HEARTBEAT_PORT=4000
HEARTBEAT_INTERVAL=10

MODEL_CHUNK_SIZE=8192*1024*1024

BLOCKCHAIN_LEADER_HOSTNAME="bc_node_0"
BLOCKCHAIN_PORTS = {
    "backend_port": 5000,
    "leader_checkin_port": 8000,
    "broadcast_port": 7000
}

NODE_REGISTRY = {
    "0": "hospital_0",
    "1": "hospital_0",
    "2": "hospital_1",
}
