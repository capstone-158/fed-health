'''
Improvement 1: a better ensemble layer than the paper's Algorithm 5.

Runs offline on the saved output probabilities of the full run (results/full_run/final_probs.npz),
so every method combines the SAME three trained global models. Weights / parameters are fitted on
the validation split (7,180) and evaluated once on the test split (10,004).

    python3 improvements/compare_ensembles.py      (from the repo root)

Methods:
    paper       one weight per hospital, 0.1-step grid search (Alg. 5, shared/ensemble.py)
    equal       1/3 each
    calibrated  per-hospital temperature scaling (fitted on val), then the paper's grid search
    per_class   one weight per hospital per class, grid search per class (coordinate ascent)
    stacking    multinomial logistic regression on the 27 log-probabilities
    one_se      paper's grid, but pick the most balanced weights within one standard error of the best
    log_loss    paper's grid, but pick the weights with the lowest val log-loss instead of highest accuracy
'''
import itertools
import json
import logging
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from shared.ensemble import grid_search_weights, STEPS  # noqa: E402
logging.disable(logging.INFO)   # grid search logs every call

HOSPITALS = ["hospital_0", "hospital_1", "hospital_2"]
EPS = 1e-12

d = np.load(os.path.join(REPO, "results", "full_run", "final_probs.npz"))
y_val, y_test = d["y_val"], d["y_test"]
val = [d[f"val_{h}"] for h in HOSPITALS]
test = [d[f"test_{h}"] for h in HOSPITALS]
n_classes = val[0].shape[1]


def acc(p, y):
    return float((p.argmax(1) == y).mean())


def combine(probs, w):
    # w: (hospitals,) or (hospitals, classes)
    w = np.asarray(w)
    if w.ndim == 1:
        return sum(wi * p for wi, p in zip(w, probs))
    return sum(w[i][None, :] * p for i, p in enumerate(probs))


# ---------------- methods ----------------

def paper(val, test):
    w, _ = grid_search_weights(val, y_val)
    return combine(test, w), {"weights": [round(x, 4) for x in w]}


def equal(val, test):
    w = [1 / len(val)] * len(val)
    return combine(test, w), {"weights": w}


def temperature(p, y):
    # softmax(log p / T) == softmax(logits / T), so we can rescale the saved probabilities directly
    logp = np.log(p + EPS)
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.05), np.log(10), 200)):
        z = logp / t
        z -= z.max(1, keepdims=True)
        q = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        nll = -np.log(q[np.arange(len(y)), y] + EPS).mean()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def apply_t(p, t):
    z = np.log(p + EPS) / t
    z -= z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def calibrated(val, test):
    ts = [temperature(p, y_val) for p in val]
    cv, ct = [apply_t(p, t) for p, t in zip(val, ts)], [apply_t(p, t) for p, t in zip(test, ts)]
    w, _ = grid_search_weights(cv, y_val)
    return combine(ct, w), {"temperatures": [round(t, 3) for t in ts], "weights": [round(x, 4) for x in w]}


def per_class(val, test, passes=2):
    # start from the paper's weights for every class, then improve one class column at a time
    w0, _ = grid_search_weights(val, y_val)
    W = np.tile(np.asarray(w0)[:, None], (1, n_classes))
    combos = [np.asarray(a) / sum(a) for a in itertools.product(STEPS, repeat=len(val)) if sum(a) > 0]
    best = acc(combine(val, W), y_val)
    for _ in range(passes):
        for c in range(n_classes):
            for a in combos:
                trial = W.copy()
                trial[:, c] = a
                s = acc(combine(val, trial), y_val)
                if s > best:
                    best, W = s, trial
    return combine(test, W), {"weights": np.round(W, 3).tolist()}


def one_se(val, test, y=None):
    # one-standard-error rule: every weight combination whose val accuracy is within 1 SE of the best
    # is "as good as the best" given the val set's size; among those take the most balanced (max entropy).
    # robust when the tuning set differs from deployment data, still drops a model that is really worse.
    y = y_val if y is None else y
    combos = [np.asarray(a) / sum(a) for a in itertools.product(STEPS, repeat=len(val)) if sum(a) > 0]
    accs = np.array([acc(combine(val, a), y) for a in combos])
    best = accs.max()
    se = np.sqrt(best * (1 - best) / len(y))
    ok = [a for a, s in zip(combos, accs) if s >= best - se]
    w = max(ok, key=lambda a: -(a * np.log(a + EPS)).sum())
    return combine(test, w), {"weights": [round(float(x), 4) for x in w], "best_val": 100 * best, "one_se": 100 * se}


def log_loss_grid(val, test):
    # same 0.1-step grid as the paper, but pick the weights with the lowest val log-loss (cross-entropy)
    # instead of the highest val accuracy: confident wrong answers are punished, and the objective is smooth
    combos = [np.asarray(a) / sum(a) for a in itertools.product(STEPS, repeat=len(val)) if sum(a) > 0]
    nll = [-np.log(combine(val, a)[np.arange(len(y_val)), y_val] + EPS).mean() for a in combos]
    w = combos[int(np.argmin(nll))]
    return combine(test, w), {"weights": [round(float(x), 4) for x in w], "val_log_loss": float(min(nll))}


def stacking(val, test):
    xv = np.hstack([np.log(p + EPS) for p in val])
    xt = np.hstack([np.log(p + EPS) for p in test])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(xv, y_val)
    return clf.predict_proba(xt), {}


METHODS = {"paper": paper, "equal": equal, "calibrated": calibrated, "per_class": per_class,
           "stacking": stacking, "one_se": one_se, "log_loss": log_loss_grid}


def paired_bootstrap(pred_a, pred_b, y, n=2000, seed=0):
    # 95% CI of (accuracy B - accuracy A) on the same test images
    rng = np.random.default_rng(seed)
    ca, cb = (pred_a == y), (pred_b == y)
    diffs = [cb[i].mean() - ca[i].mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return 100 * np.percentile(diffs, 2.5), 100 * np.percentile(diffs, 97.5)


if __name__ == "__main__":
    results, preds = {}, {}
    best_single = max(100 * acc(p, y_test) for p in test)
    for name, fn in METHODS.items():
        p_test, info = fn(val, test)
        preds[name] = p_test.argmax(1)
        results[name] = {"test_acc": 100 * acc(p_test, y_test),
                         "test_macro_f1": 100 * f1_score(y_test, preds[name], average="macro"), **info}

    print(f"\nbest single hospital on test: {best_single:.2f}%\n")
    print(f"{'method':<12}{'test acc':>10}{'macro F1':>10}{'vs paper':>10}   95% CI of difference vs paper")
    for name, r in results.items():
        lo, hi = paired_bootstrap(preds["paper"], preds[name], y_test)
        diff = r["test_acc"] - results["paper"]["test_acc"]
        r["diff_vs_paper"], r["ci95"] = diff, [lo, hi]
        print(f"{name:<12}{r['test_acc']:>9.2f}%{r['test_macro_f1']:>9.2f}%{diff:>+9.2f}%   [{lo:+.2f}, {hi:+.2f}]")

    json.dump(results, open(os.path.join(REPO, "results", "full_run", "ensemble_comparison.json"), "w"), indent=1)
    print("\nsaved results/full_run/ensemble_comparison.json")
    for name in ["calibrated", "per_class"]:
        print(name, {k: v for k, v in results[name].items() if k in ("temperatures", "weights")})
