import copy
import numpy as np
import pytest
from shared.ensemble import accuracy, grid_search_weights, result_hash, tune_ensemble


def test_accuracy():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    assert accuracy(probs, np.array([0, 1, 1])) == pytest.approx(2 / 3)


def test_grid_search_prefers_better_hospital():
    labels = np.array([0, 1, 0, 1])
    good = np.eye(2)[labels]        # always right
    bad = np.eye(2)[1 - labels]     # always wrong
    alpha, acc = grid_search_weights([good, bad], labels)
    assert acc == 1.0
    assert alpha[0] > alpha[1]
    assert sum(alpha) == pytest.approx(1.0)


def test_result_hash_is_order_independent():
    # every bc_node must get the same hash for the same result (consensus)
    assert result_hash({"a": 1, "b": 2}) == result_hash({"b": 2, "a": 1})




@pytest.fixture
def submissions():
    # Each hospital misses a different validation sample. A balanced ensemble
    # gets all three right, while every single hospital scores only 2/3.
    # On test, hospitals 0/1 get 50%, hospital 2 gets 100%.
    return {f"hospital_{i}": {
        "global_model_id": f"model-{i}",
        "val": {"probs": [([0.4, 0.6] if label == 0 else [0.6, 0.4]) if j == i
                          else ([0.9, 0.1] if label == 0 else [0.1, 0.9])
                          for j, label in enumerate([0, 1, 0])], "labels": [0, 1, 0]},
        "test": {"probs": ([[0.9, 0.1], [0.9, 0.1]] if i < 2 else [[0.9, 0.1], [0.1, 0.9]]),
                 "labels": [0, 1]},
    } for i in range(3)}


def test_three_hospital_workflow(submissions):
    result = tune_ensemble(submissions)
    assert result["type"] == "ENSEMBLE_WEIGHTS"
    assert result["global_model_ids"] == {h: s["global_model_id"] for h, s in submissions.items()}
    weights = result["weights"]
    assert set(weights) == set(submissions)
    assert sum(weights.values()) == pytest.approx(1, abs=2e-6)
    assert all(w >= 0 for w in weights.values())
    assert result["val_accuracy"] == 100
    assert all(accuracy(np.array(s["val"]["probs"]), np.array(s["val"]["labels"])) == pytest.approx(2 / 3)
               for s in submissions.values())
    assert result["individual_test_accuracy"] == {"hospital_0": 50, "hospital_1": 50, "hospital_2": 100}
    # Independently calculate the test prediction from returned weights.
    p0 = sum(weights.values()) * 0.9
    p1_class0 = 0.9 * (weights["hospital_0"] + weights["hospital_1"]) + 0.1 * weights["hospital_2"]
    expected = 50 * (int(p0 > 0.5) + int(p1_class0 < 0.5))
    assert result["test_accuracy"] == expected


@pytest.mark.parametrize("change", ["probs", "labels"])
def test_test_data_cannot_change_selected_weights(submissions, change):
    before = tune_ensemble(submissions)
    for s in submissions.values():
        s["test"][change] = [[0.1, 0.9], [0.9, 0.1]] if change == "probs" else [1, 0]
    after = tune_ensemble(submissions)
    assert after["weights"] == before["weights"]
    assert after["val_accuracy"] == before["val_accuracy"]


def test_submission_order_does_not_change_result_or_hash(submissions):
    a = tune_ensemble(submissions)
    b = tune_ensemble(dict(reversed(list(submissions.items()))))
    assert a == b
    assert result_hash(a) == result_hash(b)


@pytest.mark.parametrize("field", ["weights", "global_model_ids"])
def test_result_hash_changes_with_consensus_payload(submissions, field):
    original = tune_ensemble(submissions)
    changed = copy.deepcopy(original)
    changed[field]["hospital_0"] = 0.123 if field == "weights" else "replacement-model"
    assert result_hash(changed) != result_hash(original)


def test_empty_submissions_rejected():
    with pytest.raises(ValueError, match="at least one hospital"):
        tune_ensemble({})


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("bad, message", [
    ([], "shape"), ([[0.5, 0.5]], "shape"), ([[0.2, 0.3, 0.5]] * 3, "shapes"),
    ([[float("nan"), 0.5]] * 3, "finite"),
    ([[float("inf"), 0.5]] * 3, "finite"),
    ([[-float("inf"), 0.5]] * 3, "finite"),
    ([[0.2, 0.2]] * 3, "sum to 1"),
])
def test_invalid_probabilities_rejected(submissions, split, bad, message):
    if len(bad) == 3 and split == "test":
        bad = bad[:2]
    submissions["hospital_1"][split]["probs"] = bad
    with pytest.raises(ValueError, match=message):
        tune_ensemble(submissions)


@pytest.mark.parametrize("split", ["val", "test"])
def test_mismatched_evaluation_labels_rejected(submissions, split):
    submissions["hospital_1"][split]["labels"] = [1, 0]
    with pytest.raises(ValueError, match="different data"):
        tune_ensemble(submissions)


def test_empty_grid_search_rejected():
    with pytest.raises(ValueError, match="at least one probability"):
        grid_search_weights([], np.array([0, 1]))


@pytest.mark.parametrize("labels, message", [([], "nonempty"), ([0.5, 1.], "integers"),
    ([-1, 0], "nonnegative"), ([0, 2], "number of classes"),
    ([float("nan"), 0], "integers"), ([float("inf"), 0], "integers")])
def test_invalid_labels_rejected(submissions, labels, message):
    for s in submissions.values():
        s["test"]["labels"] = labels
    with pytest.raises(ValueError, match=message):
        tune_ensemble(submissions)
