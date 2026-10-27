import base64
import json
import numpy as np
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shared.ensemble import accuracy, grid_search_weights, result_hash
from shared.commons import verify_certificate


def test_accuracy():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    assert accuracy(probs, np.array([0, 1, 1])) == pytest.approx(2 / 3)


def test_grid_search_prefers_better_hospital():
    labels = np.array([0, 1, 0, 1])
    good = np.eye(2)[labels]        # always right
    bad = np.eye(2)[1 - labels]     # always wrong
    alpha, acc = grid_search_weights([good, bad], labels)
    assert acc == 1.0
    assert alpha[0] > alpha[1]
    assert sum(alpha) == pytest.approx(1.0)


def test_result_hash_is_order_independent():
    # every bc_node must get the same hash for the same result (consensus)
    assert result_hash({"a": 1, "b": 2}) == result_hash({"b": 2, "a": 1})


def _signed_cert(key, cert):
    data = json.dumps(cert, sort_keys=True, separators=(",", ":")).encode()
    return {**cert, "signature": base64.b64encode(key.sign(data)).decode()}


def _pem(key):
    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def test_valid_certificate_accepted():
    key = Ed25519PrivateKey.generate()
    cert = _signed_cert(key, {"node_id": "node_1", "hospital_id": "hospital_0"})
    assert verify_certificate(cert, _pem(key))


def test_tampered_certificate_rejected():
    key = Ed25519PrivateKey.generate()
    cert = _signed_cert(key, {"node_id": "node_1", "hospital_id": "hospital_0"})
    cert["hospital_id"] = "hospital_2"
    with pytest.raises(InvalidSignature):
        verify_certificate(cert, _pem(key))


def test_certificate_from_wrong_key_rejected():
    cert = _signed_cert(Ed25519PrivateKey.generate(), {"node_id": "node_1"})
    with pytest.raises(InvalidSignature):
        verify_certificate(cert, _pem(Ed25519PrivateKey.generate()))
