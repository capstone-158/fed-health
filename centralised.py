'''
centralised baseline: alexnet, resnet and net each trained alone on the whole training set
(all 6 node shards + val, like the paper), same settings as the FL runs, tested on test.pt

    python3 centralised.py --dataset colon
    python3 centralised.py --dataset breast
'''
import argparse
import json
import os
import time
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from shared.models import AdaptedAlexNet, AdaptedResNet18, ImprovedNet
from shared.config import BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY

EPOCHS = 20
DATASETS = {"colon": ("data", 9), "breast": ("data/breast", 2)}
MODELS = {"alexnet": AdaptedAlexNet, "resnet": AdaptedResNet18, "net": ImprovedNet}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUT = "results/centralised"


def load(path):
    # mmap so the full colon set (~11 GB) isnt loaded into RAM at once
    d = torch.load(path, mmap=True)
    return TensorDataset(d["images"], d["labels"].view(-1).long())


def train(model, data):
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        start, total_loss = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  epoch {epoch} loss {total_loss / len(loader):.4f} ({time.time() - start:.0f}s)", flush=True)


def evaluate(model, data):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(data, batch_size=256):
            preds.append(model(x.to(DEVICE)).argmax(1).cpu())
            labels.append(y)
    y, p = torch.cat(labels).numpy(), torch.cat(preds).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": 100 * accuracy_score(y, p), "precision": 100 * precision, "recall": 100 * recall, "f1": 100 * f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    dataset = parser.parse_args().dataset

    torch.manual_seed(42)
    data_dir, num_classes = DATASETS[dataset]
    train_data = ConcatDataset([load(f"{data_dir}/node_{i}.pt") for i in range(6)] + [load(f"{data_dir}/val.pt")])
    test_data = load(f"{data_dir}/test.pt")
    print(f"{dataset}: training on {len(train_data)} images, testing on {len(test_data)}, device {DEVICE}")

    os.makedirs(OUT, exist_ok=True)
    results = {}
    for name, model_class in MODELS.items():
        print(f"\n===== {name} =====")
        model = model_class(num_classes).to(DEVICE)
        train(model, train_data)
        results[name] = evaluate(model, test_data)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"{OUT}/{dataset}_{name}.pt")

    json.dump(results, open(f"{OUT}/{dataset}.json", "w"), indent=1)
    print(f"\n{'model':<10}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}")
    for name, r in results.items():
        print(f"{name:<10}{r['accuracy']:>9.2f}%{r['precision']:>10.2f}%{r['recall']:>8.2f}%{r['f1']:>7.2f}%")
