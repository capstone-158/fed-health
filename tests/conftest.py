import os
import sys
import tempfile

# repo root on the path so `shared`, `bc_node`, `h_node` import like they do in the containers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bc_node/ledger.py creates its ledgers at import time, keep them out of the real ledger dir
os.environ["LEDGER_PATH"] = tempfile.mkdtemp()
