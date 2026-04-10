import torch
import time
from node.training.ml_utils import get_colon_data, evaluate_model
from node.training.train_models import AdaptedAlexNet, AdaptedResNet18
from shared.config import DEVICE
import os

# Terminal color codes for fancy output
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARN = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

print(f"\n{C.BOLD}{C.CYAN}======================================================================{C.END}")
print(f"{C.BOLD}{C.CYAN}             ⚕️   PathMNIST Federated Model Evaluation                {C.END}")
print(f"{C.BOLD}{C.CYAN}======================================================================{C.END}\n")

# Use device from shared config to prevent PyTorch tensor mismatch errors
print(f"{C.WARN}[*]{C.END} Backend initialized. Compute device set to: {C.BOLD}{C.BLUE}{DEVICE}{C.END}")

print(f"\n{C.WARN}[*]{C.END} Fetching datasets from cached tensors (PathMNIST)...")
start_time = time.time()
train_dataset, _, test_dataset = get_colon_data()
print(f"    {C.GREEN}✓{C.END} Training images loaded: {C.BOLD}{len(train_dataset):,}{C.END}")
print(f"    {C.GREEN}✓{C.END} Testing images loaded:  {C.BOLD}{len(test_dataset):,}{C.END}")

print(f"\n{C.WARN}[*]{C.END} Instantiating neural network architecture: {C.BOLD}AdaptedResNet18{C.END}")
model = AdaptedResNet18(num_classes=9).to(DEVICE)
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    {C.GREEN}✓{C.END} Trainable parameters: {C.BOLD}{model_params:,}{C.END}")

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../other/models/resnet18_full_91.74_pathmnist.pth"))
print(f"\n{C.WARN}[*]{C.END} Activating pretrained global weights...")
print(f"    {C.BLUE}↳ {path}{C.END}")

model.load_state_dict(torch.load(
    path,
    map_location=DEVICE
))
print(f"    {C.GREEN}✓{C.END} State dict synced and verified.")

print(f"\n{C.WARN}[*]{C.END} Running deep evaluation on train and test distributions...")
train_acc = evaluate_model(model, train_dataset)
test_acc = evaluate_model(model, test_dataset)
elapsed = time.time() - start_time

print(f"\n{C.BOLD}{C.GREEN}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓{C.END}")
print(f"{C.BOLD}{C.GREEN}┃{C.END}{C.BOLD}                    EVALUATION RESULTS                    {C.GREEN}┃{C.END}")
print(f"{C.BOLD}{C.GREEN}┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫{C.END}")
print(f"{C.BOLD}{C.GREEN}┃{C.END}  {C.BOLD}Total Time Elapsed:{C.END} {elapsed:05.2f}s{' '*24}{C.BOLD}{C.GREEN}┃{C.END}")
print(f"{C.BOLD}{C.GREEN}┃{C.END}  {C.BOLD}Train Set Accuracy:{C.END} {C.CYAN}{train_acc:05.2f}%{C.END} {' '*29}{C.BOLD}{C.GREEN}┃{C.END}")
print(f"{C.BOLD}{C.GREEN}┃{C.END}  {C.BOLD}Test Set Accuracy: {C.END} {C.CYAN}{test_acc:05.2f}%{C.END} {' '*29}{C.BOLD}{C.GREEN}┃{C.END}")
print(f"{C.BOLD}{C.GREEN}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛{C.END}\n")