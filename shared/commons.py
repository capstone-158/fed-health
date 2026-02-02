import hashlib
import logging
from socket import *

def gen_hash(data):
    if hasattr(data, "read"):
        content = data.read()
    else:
        content = data
    
    return hashlib.sha256(content.encode()).hexdigest()

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
    return logger


'''
functions needed:
- sign_data() using private key
- hash_data()
- verify_signature()
'''
