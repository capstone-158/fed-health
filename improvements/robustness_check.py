'''
Does the one-standard-error rule still down-weight a hospital that really is worse?
(Otherwise it would just be "use equal weights".)

Starts from the full run's three strong global models and replaces ONE hospital's outputs with:
    weak       that hospital's global model from the small run (500 images/node), a real poorly trained model
    malicious  confident but wrong probabilities (class columns shifted by one)
    random     random probabilities (a broken / noisy participant)
and compares the paper's grid search, plain equal weights, and the 1-SE rule.

    python3 improvements/robustness_check.py      (from the repo root)
'''
import hashlib
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import compare_ensembles as ce  # noqa: E402
from shared.commons import get_model  # noqa: E402

MODELS = {"hospital_0": "alexnet", "hospital_1": "resnet", "hospital_2": "net"}
SMALL_RUN_BLOCK = 15        # multi_inst block of the small GPU run
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
L = os.path.join(REPO, "node_ledgers", "bc_node_ledger_0")


def small_run_probs(h, val_images, test_images):
    mid = [b for b in json.load(open(f"{L}/multi_inst_chain.json"))
           if b["index"] == SMALL_RUN_BLOCK][0]["payload"]["global_model_ids"][h]
    mhash = [b for b in json.load(open(f"{L}/{h}_chain.json"))
             if b["payload"].get("model_id") == mid][0]["payload"]["model_hash"]
    path = os.path.join(REPO, "bc_node", "client_models", "commit", f"{mhash}.pt")
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == mhash
    model = get_model(MODELS[h]).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    out = []
    for images in (val_images, test_images):
        with torch.no_grad():
            out.append(torch.cat([F.softmax(model(images[i:i + 256].to(DEVICE)), 1).cpu()
                                  for i in range(0, len(images), 256)]).numpy())
    return out


def scenario_row(name, val, test):
    y_val, y_test = ce.y_val, ce.d["y_test"]
    row = {"scenario": name, "single_test": [round(100 * ce.acc(p, y_test), 2) for p in test]}
    for m in ["paper", "equal", "one_se", "log_loss"]:
        p, info = ce.METHODS[m](val, test)
        row[m] = {"test_acc": round(100 * ce.acc(p, y_test), 2), "weights": [round(float(w), 3) for w in info["weights"]]}
    return row


if __name__ == "__main__":
    strong_val = [ce.d[f"val_{h}"] for h in MODELS]
    strong_test = [ce.d[f"test_{h}"] for h in MODELS]
    val_images = torch.load(os.path.join(REPO, "data/val.pt"), mmap=True)["images"]
    test_images = torch.load(os.path.join(REPO, "data/test.pt"), mmap=True)["images"]
    rng = np.random.default_rng(0)

    rows = [scenario_row("all strong (full run)", strong_val, strong_test)]
    for i, h in enumerate(MODELS):
        wv, wt = small_run_probs(h, val_images, test_images)
        bad = {
            f"weak {h} ({MODELS[h]}, small run)": (wv, wt),
            f"malicious {h} (confident, wrong)": (np.roll(strong_val[i], 1, axis=1), np.roll(strong_test[i], 1, axis=1)),
            f"random {h}": (rng.dirichlet(np.ones(9), len(ce.y_val)), rng.dirichlet(np.ones(9), len(ce.d["y_test"]))),
        }
        for name, (bv, bt) in bad.items():
            v, t = list(strong_val), list(strong_test)
            v[i], t[i] = bv, bt
            rows.append(scenario_row(name, v, t))

    print(f"\n{'scenario':<44}{'singles (test %)':<24}{'paper':>22}{'equal':>22}{'1-SE rule':>22}{'log-loss':>22}")
    for r in rows:
        cells = [f"{r[m]['test_acc']:.2f}% {r[m]['weights']}" for m in ["paper", "equal", "one_se", "log_loss"]]
        print(f"{r['scenario']:<44}{str(r['single_test']):<24}" + "".join(f"{c:>34}" for c in cells))
    json.dump(rows, open(os.path.join(REPO, "results", "full_run", "robustness_check.json"), "w"), indent=1)
    print("\nsaved results/full_run/robustness_check.json")
