import base64
import copy
import json
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shared.commons import verify_certificate


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


def test_verification_preserves_certificate_and_is_repeatable():
    key = Ed25519PrivateKey.generate()
    cert = _signed_cert(key, {"node_id": "0", "nested": {"hospital": "0"}})
    original = copy.deepcopy(cert)
    assert verify_certificate(cert, _pem(key))
    assert verify_certificate(cert, _pem(key))
    assert cert == original


@pytest.mark.parametrize("signature", [None, 123, "%%%", "", base64.b64encode(b"short").decode()])
def test_malformed_signature_rejected_without_mutation(signature):
    cert = {"node_id": "0", "signature": signature}
    original = copy.deepcopy(cert)
    with pytest.raises(ValueError, match="signature"):
        verify_certificate(cert, _pem(Ed25519PrivateKey.generate()))
    assert cert == original


def test_missing_signature_rejected():
    with pytest.raises(ValueError, match="signature"):
        verify_certificate({"node_id": "0"}, _pem(Ed25519PrivateKey.generate()))


def test_invalid_public_key_rejected():
    cert = _signed_cert(Ed25519PrivateKey.generate(), {"node_id": "0"})
    with pytest.raises(ValueError):
        verify_certificate(cert, b"not a PEM key")


def test_invalid_signature_rejection_preserves_input():
    key = Ed25519PrivateKey.generate()
    cert = _signed_cert(key, {"node_id": "0"})
    signature = bytearray(base64.b64decode(cert["signature"]))
    signature[0] ^= 1
    cert["signature"] = base64.b64encode(signature).decode()
    original = copy.deepcopy(cert)
    with pytest.raises(InvalidSignature):
        verify_certificate(cert, _pem(key))
    assert cert == original
