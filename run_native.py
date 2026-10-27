import logging
import os
import subprocess
import sys
import threading
import time
import httpx
import redis

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from shared.ensemble import format_summary  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)    # hides the "HTTP Request: GET ..." spam

NODES_PER_HOSPITAL = {0: ["0", "1"], 1: ["2", "3"], 2: ["4", "5"]}
INFRA = ["redis_h0", "redis_h1", "redis_h2", "bc_node_0", "bc_node_1", "bc_node_2",
         "h_node_0", "h_node_1", "h_node_2"]
BLOCKCHAIN_URL = "http://localhost:5050"     # docker-compose maps bc_node_0 5000 -> 5050

# only these lines from the hospital, blockchain and node logs make it to this terminal,
# so you can follow a round step by step without the noise. full logs are still in the files / docker
KEY_LINES = [
    "SELECTED for round", "ATTACK MODE", "train accuracy after this round", "Uploading model",   # nodes
    "Starting round", "Twin trust", "Aggregation", "FAILED", "LOCAL FL DONE", "Ensemble:",      # hospitals
    "Model upload received", "Consensus check", "added to", "Consensus failed",                 # blockchain
    "Ensemble probs received", "ALL HOSPITALS SUBMITTED", "Ensemble consensus",
    "ERROR", "Traceback",
]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} - {msg}", flush=True)


def show(who, line):
    if not any(k in line for k in KEY_LINES):
        return
    # "2026-10-27 10:22:00,146 - INFO - msg" -> "msg"
    msg = line.split(" - ", 2)[-1].replace("!", "").strip()
    log(f"[{who}] {msg}")


def follow_docker_logs():
    proc = subprocess.Popen(["docker", "compose", "logs", "-f", "--no-color", "h_node_0", "h_node_1", "h_node_2", "bc_node_0"],
                            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    names = {"h_node_0": "hospital 0", "h_node_1": "hospital 1", "h_node_2": "hospital 2", "bc_node_0": "blockchain"}

    def read():
        for line in proc.stdout:
            service, _, rest = line.partition("|")
            show(names.get(service.strip().rsplit("-", 1)[0], service.strip()), rest.strip())

    threading.Thread(target=read, daemon=True).start()
    return proc


def follow_node_log(node_id, path):
    def read():
        open(path, "a").close()
        with open(path) as f:
            f.seek(0, os.SEEK_END)     # older runs are in this file too, only show the new lines
            while True:
                line = f.readline()
                if line:
                    show(f"node {node_id}", line.strip())
                else:
                    time.sleep(1)

    threading.Thread(target=read, daemon=True).start()


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
    for key in ["ROUNDS", "EPOCHS", "BATCH_SIZE", "MICRO_BATCH", "NUM_CLASSES", "ATTACK", "ATTACK_NODE"]:
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

    log("Following hospital, blockchain and node logs, key steps show up below as [who] what")
    docker_logs = follow_docker_logs()
    os.makedirs(os.path.join(REPO, "native_run", "logs"), exist_ok=True)
    for node_ids in NODES_PER_HOSPITAL.values():
        for n in node_ids:
            follow_node_log(n, os.path.join(REPO, "native_run", "logs", f"node_{n}.log"))

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

    time.sleep(2)    # let the last log lines through before the table
    docker_logs.terminate()
    print(format_summary(block["payload"], block["index"], block["block_hash"]))
    log(f"Total run time {(time.time() - run_start) / 3600:.2f} h. Docker services are still running, "
        f"stop them with: docker compose down")


if __name__ == "__main__":
    main()
