# How to run
Prerequisites:
```bash
brew install --cask orbstack
brew install docker

# In the main project directory
mkdir data/ models/
git pull origin main
```

for macos:
```bash
orb start
docker compose up --build
```

## Running the Project After the Initial Build

Once the project has been built successfully for the first time, you do not need to rebuild it every time.

Just do:

```bash
orb start
docker compose up -d
```

If `Dockerfile` or `requirements.txt` is modified, then rebuild the images using:

```bash
docker compose up --build
```

todo: check TODO marker


## Automated tests

The six test modules use synthetic tensors, temporary storage, in-process HTTP
requests, and fake Redis. They do not train CNNs, download datasets or pretrained
weights, or start Docker services.

Use Python 3.11 with the runtime dependencies installed, then:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The runtime requirements in `node/`, `h_node/`, and `bc_node/` target Linux CPU
containers. For a native macOS test environment, use a compatible PyTorch and
torchvision pair from the standard package index rather than the `+cpu` Linux
pins. For example, the suite was verified using the existing PyTorch 2.9.0 with
torchvision 0.24.0 in a temporary environment:

```bash
python3.11 -m venv --system-site-packages /tmp/fed-health-tests-venv
/tmp/fed-health-tests-venv/bin/python -m pip install -r requirements-test.txt \
  'torch==2.9.0' 'torchvision==0.24.0' numpy cryptography redis requests \
  httpx fastapi uvicorn ping3 medmnist
/tmp/fed-health-tests-venv/bin/python -m pytest -q
```

`tests/conftest.py` redirects import-time ledger initialization to a managed
temporary directory. Each ledger test also uses isolated per-test storage.
Server modules expose their existing handlers without launching background
services when imported; their normal script entry points still launch services.

New ledgers use deterministic genesis blocks. Existing valid ledgers are loaded
without rewriting their genesis blocks; already-diverged historical replicas
require a separate reconciliation process. Hash validation detects modified
blocks and broken links, but does not authenticate a completely rewritten chain.

### Deployment compatibility review (2026-10-27)

The original 104 tests plus one new integration smoke test passed in both the
native environment and an isolated Linux ARM64 image: 105 passed, 0 failed,
0 skipped. The Docker test image extends the cached project edge-node image,
installs the union of all three declared runtime requirements and test
requirements, and uses Python 3.11.16, torch 2.11.0+cpu, torchvision 0.26.0+cpu,
and pytest 8.4.2. Tests ran with networking disabled, the repository mounted
read-only, and temporary writable storage; production Compose volumes were not
used. Unpinned dependencies can resolve differently in future builds.

The integration smoke test submits six synthetic edge checkpoints, verifies
hash agreement at two simulated peers, downloads verified models for three
hospital FedAvg operations, records the global models, runs synthetic inference,
and commits the consensus ensemble result. It checks all three replica chains
and nine committed models per replica. It uses real application handlers and
numerical code with simulated peer/HTTP transport, so it does not verify live
TCP framing, Redis services, container startup orchestration, or CNN training.

Additional review checks compared 40 ensemble cases (including hashes) and 40
FedAvg cases exactly against the pre-change Git HEAD implementation in each
environment. All 73 nonempty existing tensor files loaded, with 72 checkpoints
strictly compatible with the unchanged TinyCNN. All 12 historical ledger copies
validated. Before/after SHA-256 comparisons confirmed all 87 existing artifacts
were unchanged. Two pre-existing empty blockchain staging files remain empty
and unreadable; they are not committed checkpoints and were left untouched.
