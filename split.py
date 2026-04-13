import os
import torch
from torch.utils.data import random_split
from torchvision import transforms
from medmnist import PathMNIST

DATA_DIR = "data"


def split_pathmnist(train_dataset, num_hospitals=3, nodes_per_hospital=2):
    total_len = len(train_dataset)

    # hospital level split
    hospital_size = total_len // num_hospitals
    hospital_lengths = [hospital_size] * num_hospitals
    hospital_lengths[-1] += total_len - sum(hospital_lengths)

    hospital_datasets = random_split(
        train_dataset,
        hospital_lengths,
        generator=torch.Generator().manual_seed(42)
    )

    # node level split inside each hospital
    hospitals_nodes = []

    for h_data in hospital_datasets:
        h_len = len(h_data)

        node_size = h_len // nodes_per_hospital
        node_lengths = [node_size] * nodes_per_hospital
        node_lengths[-1] += h_len - sum(node_lengths)

        node_datasets = random_split(
            h_data,
            node_lengths,
            generator=torch.Generator().manual_seed(42)
        )

        hospitals_nodes.append(node_datasets)

    return hospitals_nodes


def save_node_dataset(node_dataset, save_path):
    images = []
    labels = []

    for img, label in node_dataset:
        images.append(img)
        labels.append(torch.tensor(label).long())

    images = torch.stack(images)
    labels = torch.stack(labels)

    torch.save({
        "images": images,
        "labels": labels
    }, save_path)

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    print(f"Saved: {save_path}")
    print(f"  Number of images: {len(node_dataset)}")
    print(f"  File size: {file_size_mb:.2f} MB")


transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train = PathMNIST(split="train", transform=transform, download=False)
val   = PathMNIST(split="val", transform=transform, download=False)
test  = PathMNIST(split="test", transform=transform, download=False)

print(f"\nOriginal Dataset Sizes:")
print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")

hospitals_nodes = split_pathmnist(
    train,
    num_hospitals=3,
    nodes_per_hospital=2
)

os.makedirs(DATA_DIR, exist_ok=True)

print("\nSaving node datasets:\n")

for i, hospital in enumerate(hospitals_nodes):
    for j, node in enumerate(hospital):
        file_name = f"h{i+1}n{j+1}.pt"
        save_path = os.path.join(DATA_DIR, file_name)
        save_node_dataset(node, save_path)