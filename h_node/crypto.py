from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from datetime import datetime, timedelta
import base64
import json


def generate_keys():
    private_key = ed25519.Ed25519PrivateKey.generate()

    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return private_key, private_pem, public_pem

def issue_node_certificate(hospital_id, node_id, private_key):
    certificate = {
        "node_id": node_id,
        "hospital_id": f"hospital_{hospital_id}",
        "issued_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(days=365)).isoformat(),
    }

    cert_bytes = json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode('utf-8')
    signature = private_key.sign(cert_bytes)

    certificate["signature"] = base64.b64encode(signature).decode("utf-8")

    return certificate

