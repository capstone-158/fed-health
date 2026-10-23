import torch
import logging
import requests
from medmnist import PathMNIST
from torchvision import transforms
import torch.nn.functional as F
import itertools
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
import sys
import os
from shared.models import AdaptedAlexNet, AdaptedResNet18, ImprovedNet
from shared.state_manager import get_hospital_current_round
from shared.config import BATCH_SIZE, MICRO_BATCH, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, EPOCHS, NUM_CLASSES, HOSPITAL_URL
from shared.commons import get_model, get_logger

logger = get_logger()
# mps = Apple GPU, only available when the node runs natively on macOS (docker on mac is cpu only)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

from torchvision import transforms
from medmnist import PathMNIST

# TODO: get mounted data and transform the subset of data
# node, might be useless now
def get_colon_data():
    transform = transforms.Compose([
        transforms.Resize((100, 100), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = PathMNIST(split='train', transform=transform, download=True)
    val_dataset = PathMNIST(split='val', transform=transform, download=True)
    test_dataset = PathMNIST(split='test', transform=transform, download=True)

    return train_dataset, val_dataset, test_dataset


# ensemble weight grid search (Alg. 5) + ensemble eval moved to shared/ensemble.py,
# it now runs on the blockchain nodes over the hospitals' output probabilities

'''
remove once everything's working

def get_model():
    model_type = os.getenv("model_type")
    if model_type == "resnet":
        return AdaptedResNet18(NUM_CLASSES)
    elif model_type == "alexnet":
        return AdaptedAlexNet(NUM_CLASSES)
    elif model_type == "net":
        return ImprovedNet(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_global_weights(model_type):
    round_num = get_hospital_current_round()

    if round_num == 0:
        model = get_model(model_type)
        return model.state_dict()
    else:
        if not os.path.exists(GLOBAL_WEIGHTS_PATH):
            raise FileNotFoundError(f"{GLOBAL_WEIGHTS_PATH} not found")
        return torch.load(GLOBAL_WEIGHTS_PATH, map_location="cpu")
'''
class NodeDataset(Dataset):
    def __init__(self, data_path):
        # mmap: shard stays on disk / page cache instead of fully in RAM (shards are up to 1.8 GB)
        data = torch.load(data_path, map_location="cpu", mmap=True)
        self.images = data["images"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def client_update(data_path, weights_path, model_path, on_epoch=None):
    # on_epoch(epoch): called at the start of every epoch so the heartbeat can report live progress
    endpoint = "/get_model_type"
    res = requests.get(HOSPITAL_URL + endpoint)
    res.raise_for_status()
    model_type = res.json().get("model")

    model = get_model(model_type).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    client_dataset = NodeDataset(data_path)
    train_loader = DataLoader(client_dataset, batch_size=MICRO_BATCH, shuffle=True)
    accum_steps = max(1, BATCH_SIZE // MICRO_BATCH)
    logger.info("Training loop: Data loaded")

    logger.info(f"Training loop: Starting model training now, batch {BATCH_SIZE} as {accum_steps} x {MICRO_BATCH}")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    metrics = {
        "loss": [],
        "accuracy": []
    }

    for epoch in range(EPOCHS):
        # back to train mode each epoch: evaluate_model() flips the model to eval()
        # at the end of the previous epoch, which would otherwise leave BatchNorm/Dropout
        # in inference mode for the rest of training.
        model.train()
        if on_epoch:
            on_epoch(epoch)
        logger.info(f"Training loop: Node {os.getenv('NODE_ID')}, currently on epoch {epoch}")
        total_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.view(-1).long().to(DEVICE)

            outputs = model(x)
            loss = criterion(outputs, y)
            # scale so the summed gradient over accum_steps micro batches equals one big batch's
            (loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}")

        avg_loss = total_loss / len(train_loader)
        metrics["loss"].append(avg_loss)
        logger.info(f"Training loop: Node {os.getenv('NODE_ID')}, epoch {epoch} loss {avg_loss:.4f}")

    # train accuracy once per round, not every epoch: it's a full extra pass over the shard
    train_acc = evaluate_model(model, client_dataset)
    metrics["accuracy"].append(train_acc)
    logger.info(f"Training loop: Node {os.getenv('NODE_ID')}, train accuracy after this round {train_acc:.2f}%")
    metrics["dataset_len"] = len(client_dataset)

    logger.info("Training loop: Training complete")
    # save cpu tensors so the hospital (linux container, no gpu) can load it
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    logger.info(f"Model saved to {model_path}")

    return metrics

def evaluate_model(model, dataset):
    loader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.squeeze().long().to(DEVICE)

            outputs = model(x)
            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100.0 * correct / total