'''
Ensemble weight tuning (paper Algorithm 5) + ensemble evaluation.

Works purely on output probabilities P_Hi, never on the models themselves, so it
can run on the blockchain nodes (B_M). Every B_M node runs tune_ensemble() on the
same inputs; the hash of the result is what gets compared for consensus (Alg. 6).
'''
import hashlib
import itertools
import json
import numpy as np
from shared.commons import get_logger

logger = get_logger()

# grid step of 0.1, same as the original grid search
STEPS = [round(0.1 * i, 1) for i in range(11)]


def validate_predictions(probs_list, labels, context="predictions"):
    """Reject malformed evaluation data before optimization or scoring."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not labels.size:
        raise ValueError(f"{context}: labels must be a nonempty 1D array")
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError(f"{context}: labels must be nonnegative integers")
    if not probs_list:
        raise ValueError(f"{context}: at least one probability array is required")
    shape = None
    for probs in probs_list:
        probs = np.asarray(probs)
        if probs.ndim != 2 or probs.shape[0] != len(labels) or probs.shape[1] < 2:
            raise ValueError(f"{context}: probabilities must have shape (samples, classes)")
        if shape is not None and probs.shape != shape:
            raise ValueError(f"{context}: incompatible probability shapes")
        shape = probs.shape
        if not np.issubdtype(probs.dtype, np.number) or not np.isfinite(probs).all():
            raise ValueError(f"{context}: probabilities must contain finite numeric values")
        if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1):
            raise ValueError(f"{context}: probability rows must be nonnegative and sum to 1")
        if (labels >= probs.shape[1]).any():
            raise ValueError(f"{context}: labels exceed the number of classes")


def accuracy(probs, labels):
    return float((np.argmax(probs, axis=1) == labels).mean())


def grid_search_weights(probs_list, labels):
    '''
    probs_list: one (N, C) array per hospital, rows aligned to the same samples
    labels: (N,) true labels
    returns (best_alpha, best_acc), alpha is l1-normalised
    '''
    validate_predictions(probs_list, labels)
    h = len(probs_list)
    logger.info(f"Grid search: testing {len(STEPS) ** h} weight combinations for {h} hospitals over {len(labels)} samples")
    best_acc = 0.0
    best_alpha = tuple([1.0 / h] * h)   # default equal weights

    for alpha in itertools.product(STEPS, repeat=h):
        total = sum(alpha)
        if total == 0:      # skip all-zero combination
            continue
        alpha = tuple(a / total for a in alpha)

        # P~(alpha) = sum_i alpha_i * P_Hi   (Eq. 4)
        p_ensemble = sum(a * p for a, p in zip(alpha, probs_list))
        acc = accuracy(p_ensemble, labels)

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    logger.info(f"Grid search: best weights found {best_alpha}, val accuracy {best_acc*100:.2f}%")
    return best_alpha, best_acc


def tune_ensemble(submissions):
    '''
    submissions: {hospital_id: {"global_model_id", "val": {"probs", "labels"}, "test": {...}}}

    Tunes the weights on the validation split, then reports the test accuracy of
    the weighted ensemble and of each hospital's global model on its own.
    '''
    if not submissions:
        raise ValueError("submissions must contain at least one hospital")
    hospitals = sorted(submissions)
    val_labels = np.array(submissions[hospitals[0]]["val"]["labels"])
    test_labels = np.array(submissions[hospitals[0]]["test"]["labels"])

    for h in hospitals:
        if submissions[h]["val"]["labels"] != submissions[hospitals[0]]["val"]["labels"] \
                or submissions[h]["test"]["labels"] != submissions[hospitals[0]]["test"]["labels"]:
            raise ValueError(f"{h} evaluated on different data than {hospitals[0]}")

    val_probs = [np.array(submissions[h]["val"]["probs"]) for h in hospitals]
    test_probs = [np.array(submissions[h]["test"]["probs"]) for h in hospitals]

    logger.info(f"Tuning ensemble for {hospitals}, global model ids: {[submissions[h]['global_model_id'] for h in hospitals]}")
    validate_predictions(val_probs, val_labels, "validation")
    validate_predictions(test_probs, test_labels, "test")
    if val_probs[0].shape[1] != test_probs[0].shape[1]:
        raise ValueError("validation and test probability shapes must use the same classes")
    alpha, val_acc = grid_search_weights(val_probs, val_labels)
    test_ensemble = sum(a * p for a, p in zip(alpha, test_probs))
    logger.info(f"Ensemble test accuracy with these weights: {100 * accuracy(test_ensemble, test_labels):.2f}%")

    return {
        "type": "ENSEMBLE_WEIGHTS",
        "global_model_ids": {h: submissions[h]["global_model_id"] for h in hospitals},
        "weights": {h: round(a, 6) for h, a in zip(hospitals, alpha)},
        "val_accuracy": round(100 * val_acc, 4),
        "test_accuracy": round(100 * accuracy(test_ensemble, test_labels), 4),
        "individual_test_accuracy": {
            h: round(100 * accuracy(p, test_labels), 4) for h, p in zip(hospitals, test_probs)
        },
    }


def result_hash(result):
    # h_i = generate_hash(alpha, GM_id) in Alg. 6
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def format_summary(result, block_index=None, block_hash=None, agreed=None):
    # neat end-of-run table, used in the leader's (and last hospital's) logs
    line = "=" * 78
    sep = "-" * 78
    out = ["", line, "FINAL RESULTS  -  hierarchical ensemble FL".center(78), line]

    if block_index is not None:
        out.append(f" multi_inst block     : #{block_index}" + (f"  (hash {block_hash[:16]}...)" if block_hash else ""))
    if agreed is not None:
        out.append(f" consensus            : {agreed} bc_nodes produced the same weights")

    out += [sep, f" {'hospital':<12}| {'global model id':<38}| {'weight':>7} | {'test acc':>9}", sep]
    for h, w in result["weights"].items():
        out.append(f" {h:<12}| {str(result['global_model_ids'][h]):<38}| {w:>7.4f} | {result['individual_test_accuracy'][h]:>8.2f}%")
    out.append(sep)

    best_h = max(result["individual_test_accuracy"], key=result["individual_test_accuracy"].get)
    best_acc = result["individual_test_accuracy"][best_h]
    out += [
        f" {'ensemble val accuracy (used for tuning)':<62}| {result['val_accuracy']:>8.2f}%",
        f" {'ensemble test accuracy':<62}| {result['test_accuracy']:>8.2f}%",
        f" {'best single hospital (' + best_h + ')':<62}| {best_acc:>8.2f}%",
        f" {'gain of ensemble over best single hospital':<62}| {result['test_accuracy'] - best_acc:>+8.2f}%",
        line,
    ]
    return "\n".join(out)
