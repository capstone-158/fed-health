'''
functions needed:
- sign_data() using private key
- hash_data()
- verify_signature()
'''

import hashlib

def gen_hash(data):
    if hasattr(data, "read"):
        content = data.read()
    else:
        content = data
    
    return hashlib.sha256(content.encode()).hexdigest()