import torch
import math
import torch.nn.functional as F
from shared.config import BATCH_SIZE
from shared.commons import get_model, get_logger

logger = get_logger()

def perform_fedavg(client_updates, global_weights_path):
    """
    client_updates: LIST of tuples as follows
        [
            (state_dict_1, num_samples_1),
            (state_dict_2, num_samples_2),
            ...
        ]
    """

    if not client_updates:
        raise ValueError("client_updates empty")

    weights = [weight for _, weight in client_updates]
    total_samples = sum(weights)
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isfinite(total_samples) or total_samples <= 0:
        raise ValueError("aggregation weights must be finite, nonnegative, and have a positive total")
    reference = client_updates[0][0]
    for state, _ in client_updates:
        if state.keys() != reference.keys() or any(state[k].shape != reference[k].shape for k in reference):
            raise ValueError("client model keys and tensor shapes must match")

    avg_state_dict = {}
    first_state_dict = client_updates[0][0] #to know shape

    for key in first_state_dict.keys():
        # accumulate in float: BatchNorm's num_batches_tracked is an int tensor
        avg_state_dict[key] = torch.zeros_like(first_state_dict[key], dtype=torch.float32) #same shape filled with zeros

    for state_dict, num_samples in client_updates:
        weight = num_samples / total_samples #how much the client contributes

        for key in state_dict.keys():
            avg_state_dict[key] += state_dict[key].float() * weight

    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key].to(first_state_dict[key].dtype)

    torch.save(avg_state_dict, global_weights_path)

    return avg_state_dict #global model weights after averaging


def predict_probs(global_weights_path, data_path):
    """
    Runs this hospital's global model GM_i over a shared evaluation split and returns
    the class probabilities P_Hi (one row per sample, same order for every hospital).
    This is the step the paper does under SMPC (Alg. 2-3); here it runs in the clear.
    """
    model = get_model()
    model.load_state_dict(torch.load(global_weights_path, map_location="cpu"))
    model.eval()

    data = torch.load(data_path, map_location="cpu", mmap=True)
    images, labels = data["images"], data["labels"].view(-1).long()
    logger.info(f"Predicting probs: loaded {len(images)} images from {data_path}")

    probs = []
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            probs.append(F.softmax(model(images[i:i + BATCH_SIZE]), dim=1))

    probs = torch.cat(probs)
    acc = 100.0 * (probs.argmax(dim=1) == labels).float().mean().item()
    logger.info(f"Predicting probs: done, own accuracy of this global model on {data_path}: {acc:.2f}%")

    return probs.tolist(), labels.tolist()
