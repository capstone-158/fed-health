'''
Hybrid run: redis, hospitals (h_node) and blockchain (bc_node) run in docker, but the 6 training
nodes run natively on macOS so they can use the Apple GPU (mps). Docker on a mac is CPU only.

    python3 run_native.py envs/small.env       # quick check
    python3 run_native.py envs/full.env        # paper settings, all data

What it does:
    1. docker compose down, then up everything EXCEPT node_0..node_5
    2. waits until the hospitals and the blockchain leader answer
    3. runs the nodes one hospital at a time (both nodes of hospital 0, then 1, then 2):
       they share one GPU anyway, and this keeps only 2 trainings in memory
    4. moves on when that hospital's round counter in redis reaches ROUNDS
    5. waits for the new multi_inst block and prints the final results table

Node logs: native_run/logs/node_{id}.log     Docker logs: docker compose logs -f
Keep the Mac plugged in, lid open, and run `caffeinate -dims` in another terminal.
'''
import os
import subprocess
import sys
import time
import httpx
import redis

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from shared.ensemble import format_summary  # noqa: E402

NODES_PER_HOSPITAL = {0: ["0", "1"], 1: ["2", "3"], 2: ["4", "5"]}
INFRA = ["redis_h0", "redis_h1", "redis_h2", "bc_node_0", "bc_node_1", "bc_node_2",
         "h_node_0", "h_node_1", "h_node_2"]
BLOCKCHAIN_URL = "http://localhost:5050"     # docker-compose maps bc_node_0 5000 -> 5050


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} - {msg}", flush=True)


def load_env_file(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                env[key] = value
    return env


def wait_for(url, what, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(url, timeout=5).status_code == 200:
                log(f"{what} is up")
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise SystemExit(f"{what} did not come up at {url}")


def hospital_round(h):
    r = redis.Redis(host="localhost", port=6380 + h, decode_responses=True)
    return int(r.get(f"fl:hospital:{h}:current_round") or 0)


def start_node(node_id, h, run_env, data_dir):
    workdir = os.path.join(REPO, "native_run", f"node_{node_id}")
    os.makedirs(os.path.join(REPO, "native_run", "logs"), exist_ok=True)
    env = dict(os.environ)
    env.update({
        "NODE_ID": node_id,
        "H_ID": str(h),
        "H_HOST": "localhost",
        "HOSPITAL_URL": f"http://localhost:{8000 + 100 * h}",
        "HOSPITAL_HEARTBEAT_PORT": str(4000 + 100 * h),
        "REDIS_HOST": "localhost",
        "REDIS_PORT": str(6380 + h),
        "BLOCKCHAIN_URL": BLOCKCHAIN_URL,
        "DATA_DIR": data_dir,
        "NODE_WORKDIR": workdir,
        "PYTHONPATH": REPO,
        "PYTHONUNBUFFERED": "1",
    })
    for key in ["ROUNDS", "EPOCHS", "BATCH_SIZE", "MICRO_BATCH", "NUM_CLASSES"]:
        if key in run_env:
            env[key] = run_env[key]

    log_path = os.path.join(REPO, "native_run", "logs", f"node_{node_id}.log")
    log_file = open(log_path, "a")
    proc = subprocess.Popen([sys.executable, "server.py"], cwd=os.path.join(REPO, "node"),
                            env=env, stdout=log_file, stderr=subprocess.STDOUT)
    log(f"Started node {node_id} (hospital {h}) natively, pid {proc.pid}, log {log_path}")
    return proc


def stop(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 run_native.py envs/<mode>.env")
    env_file = sys.argv[1]
    run_env = load_env_file(env_file)
    rounds = int(run_env.get("ROUNDS", 10))
    data_dir = os.path.abspath(os.path.join(REPO, run_env["DATA_DIR"]))

    log(f"Mode {env_file}: data {data_dir}, models {run_env['H0_MODEL']}/{run_env['H1_MODEL']}/{run_env['H2_MODEL']}, "
        f"{rounds} rounds x {run_env.get('EPOCHS')} epochs, batch {run_env.get('BATCH_SIZE')}")

    # fresh docker state (redis round counters start at 0), nodes are NOT started in docker
    log("Restarting docker services without the node containers")
    subprocess.run(["docker", "compose", "down"], cwd=REPO, check=True)
    subprocess.run(["docker", "compose", "--env-file", env_file, "up", "-d", "--build", *INFRA], cwd=REPO, check=True)

    for h in range(3):
        wait_for(f"http://localhost:{8000 + 100 * h}/get_model_type", f"hospital {h}")
    wait_for(f"{BLOCKCHAIN_URL}/ledger/multi_inst", "blockchain leader")
    blocks_before = len(httpx.get(f"{BLOCKCHAIN_URL}/ledger/multi_inst", timeout=30).json())

    run_start = time.time()
    for h, node_ids in NODES_PER_HOSPITAL.items():
        log(f"===== hospital {h} ({run_env[f'H{h}_MODEL']}): training nodes {node_ids} on the GPU =====")
        hospital_start = time.time()
        procs = [start_node(n, h, run_env, data_dir) for n in node_ids]
        last_round = -1
        try:
            while hospital_round(h) < rounds:
                dead = [p for p in procs if p.poll() is not None]
                if dead:
                    raise SystemExit(f"node process {dead[0].pid} exited, check native_run/logs/")
                current = hospital_round(h)
                if current != last_round:
                    log(f"hospital {h}: on round {current} of {rounds}")
                    last_round = current
                time.sleep(30)
        finally:
            stop(procs)
        log(f"hospital {h} done in {(time.time() - hospital_start) / 60:.1f} min, nodes stopped")

    log("All hospitals trained, waiting for the ensemble block on the multi-institutional blockchain")
    while True:
        chain = httpx.get(f"{BLOCKCHAIN_URL}/ledger/multi_inst", timeout=30).json()
        if len(chain) > blocks_before:
            block = chain[-1]
            break
        time.sleep(15)

    print(format_summary(block["payload"], block["index"], block["block_hash"]))
    log(f"Total run time {(time.time() - run_start) / 3600:.2f} h. Docker services are still running, "
        f"stop them with: docker compose down")


if __name__ == "__main__":
    main()
