import hashlib
from .config import NUM_CLASSES
from .models import AdaptedAlexNet, AdaptedResNet18, ImprovedNet, TinyCNN
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
    if not isinstance(certificate, dict) or not isinstance(certificate.get("signature"), str):
        raise ValueError("certificate must contain a base64 signature")
    try:
        signature = base64.b64decode(certificate["signature"], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("certificate signature must be valid base64") from exc
    if len(signature) != 64:
        raise ValueError("certificate signature must be 64 bytes")
    unsigned_certificate = {k: v for k, v in certificate.items() if k != "signature"}

    cert_bytes = json.dumps(
        unsigned_certificate,
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")

    public_key = serialization.load_pem_public_key(public_key_pem)

    public_key.verify(signature, cert_bytes)

    return True

def get_model(model_type=None):
    if model_type is None:
        model_type = os.getenv("model_type")

    if model_type == "resnet":
        return AdaptedResNet18(NUM_CLASSES)
    elif model_type == "alexnet":
        return AdaptedAlexNet(NUM_CLASSES)
    elif model_type == "net":
        return ImprovedNet(NUM_CLASSES)
    elif model_type == "test":
        return TinyCNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")