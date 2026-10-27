'''
Digital twin trust: before FedAvg, check every node's model against what the hospital's twin saw
during the round, and keep a trust score per node that carries over between rounds.

    A  did it train?         twin saw >= half the epochs, and its weights differ from the global it started from
    B  honest about size?    claimed images / observed time per epoch can't be way faster than the slowest node
    C  is the model good?    accuracy on a held-out set, scaled between "always guess the majority class" and the best model
    D  track record          trust = 0.7 * old trust + 0.3 * this round's score
'''
import torch
from shared import state_manager
from shared.commons import get_model, get_logger
from shared.config import EPOCHS, VAL_DATA_PATH

logger = get_logger()

HOLDOUT_SIZE = 500
TRUST_DECAY = 0.7
MAX_SIZE_RATIO = 2.0

holdout = None


def load_holdout():
    global holdout
    if holdout is None:
        data = torch.load(VAL_DATA_PATH, map_location="cpu", mmap=True)
        holdout = (data["images"][:HOLDOUT_SIZE].clone(), data["labels"][:HOLDOUT_SIZE].view(-1).long().clone())
    return holdout


def holdout_accuracy(state_dict):
    images, labels = load_holdout()
    model = get_model()
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(images[i:i + 100]).argmax(1) for i in range(0, len(images), 100)])
    return (preds == labels).float().mean().item()


def weights_changed(state_dict, global_state):
    diff = sum((state_dict[k].float() - global_state[k].float()).norm() ** 2 for k in global_state)
    return diff.sqrt().item() > 1e-6


def score_round(hospital_id, round_id, updates, global_state):
    '''
    updates: [{"node_id", "state_dict", "num_samples"}] from get_round_models
    global_state: the global model this round started from
    returns the same updates with "trust" added (and num_samples capped if inflated), plus a report per node
    '''
    _, labels = load_holdout()
    chance = labels.bincount().max().item() / len(labels)
    global_acc = holdout_accuracy(global_state)

    for u in updates:
        u["twin"] = state_manager.get_twin_round(hospital_id, u["node_id"], round_id)
        u["holdout_acc"] = holdout_accuracy(u["state_dict"])
        u["trained"] = u["twin"]["epochs_seen"] >= EPOCHS / 2 and weights_changed(u["state_dict"], global_state)

    # B: lying about size only makes a node look faster, so the slowest node is the honest reference
    rates = [u["num_samples"] * u["twin"]["epochs_seen"] / u["twin"]["train_seconds"]
             for u in updates if u["trained"] and u["twin"]["train_seconds"] > 0]
    reference_rate = min(rates) if rates else None

    best_acc = max([global_acc] + [u["holdout_acc"] for u in updates])
    report = {}
    for u in updates:
        flags = []
        claimed = u["num_samples"]
        if not u["trained"]:
            flags.append("no_training")
        elif reference_rate and u["twin"]["train_seconds"] > 0:
            plausible = reference_rate * u["twin"]["train_seconds"] / u["twin"]["epochs_seen"]
            if claimed > MAX_SIZE_RATIO * plausible:
                u["num_samples"] = int(plausible)
                flags.append("size_inflated")

        quality = max(0.0, min(1.0, (u["holdout_acc"] - chance) / max(best_acc - chance, 1e-6)))
        if quality < 0.5:
            flags.append("low_quality")
        round_score = quality if u["trained"] else 0.0

        old_trust = state_manager.get_trust(hospital_id, u["node_id"])
        u["trust"] = TRUST_DECAY * old_trust + (1 - TRUST_DECAY) * round_score
        state_manager.set_trust(hospital_id, u["node_id"], u["trust"])
        # this round's score kicks a bad model out right away, trust keeps a bad history counting less
        u["weight"] = u["num_samples"] * u["trust"] * round_score

        report[u["node_id"]] = {
            "trust": round(u["trust"], 4),
            "round_score": round(round_score, 4),
            "holdout_acc": round(100 * u["holdout_acc"], 2),
            "epochs_seen": u["twin"]["epochs_seen"],
            "claimed_samples": claimed,
            "used_samples": u["num_samples"],
            "fedavg_weight": round(u["weight"], 2),
            "flags": flags,
        }
        logger.info(f"Twin trust: round {round_id} node {u['node_id']}: trust {old_trust:.3f} -> {u['trust']:.3f}, "
                    f"holdout acc {100 * u['holdout_acc']:.2f}%, epochs seen {u['twin']['epochs_seen']}, flags {flags}")

    logger.info(f"Twin trust: round {round_id} global model holdout acc {100 * global_acc:.2f}%, chance {100 * chance:.2f}%")
    return updates, report
