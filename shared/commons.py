import hashlib
from config import NUM_CLASSES
from models import AdaptedAlexNet, AdaptedResNet18, ImprovedNet
import os
import logging
from socket import *
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
import base64
import json

def create_socket(bind_addr, port):
    sock = socket(AF_INET, SOCK_STREAM)
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)    # so u dont have to wait for restart
    sock.bind((bind_addr, port))

    return sock

def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger


def verify_certificate(certificate, public_key_pem):
    signature = base64.b64decode(certificate.pop("signature"))

    cert_bytes = json.dumps(
        certificate,
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")

    public_key = serialization.load_pem_public_key(public_key_pem)

    public_key.verify(signature, cert_bytes)

    return True

def get_data_path():
    NODE_ID = os.getenv("NODE_ID")
    num = str((int(NODE_ID) % 2) + 1)

    return f"/app/shared/datasets/pathmnist_split_{num}.pt"

def get_model():
    model_type = os.getenv("model_type")
    if model_type == "resnet":
        return AdaptedResNet18(NUM_CLASSES)
    elif model_type == "alexnet":
        return AdaptedAlexNet(NUM_CLASSES)
    elif model_type == "net":
        return ImprovedNet(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")