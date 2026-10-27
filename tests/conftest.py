import os
import sys
import tempfile
import atexit
import pytest

# repo root on the path so `shared`, `bc_node`, `h_node` import like they do in the containers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# bc_node/ledger.py creates its ledgers at import time, keep them out of the real ledger dir
_ledger_dir = tempfile.TemporaryDirectory(prefix="fed-health-tests-")
atexit.register(_ledger_dir.cleanup)
os.environ["LEDGER_PATH"] = _ledger_dir.name


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    from bc_node import ledger
    monkeypatch.setattr(ledger, "LEDGER_PATH", str(tmp_path))
    return tmp_path
