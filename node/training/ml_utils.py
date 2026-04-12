import torch
from medmnist import PathMNIST
from torchvision import transforms
import torch.nn.functional as F
import itertools
from torch.utils.data import TensorDataset, DataLoader, random_split
import sys
import os
from shared.models import AdaptedAlexNet, AdaptedResNet18, ImprovedNet
from shared.state_manager import get_hospital_current_round
from shared.config import BATCH_SIZE, LEARNING_RATE, EPOCHS, NUM_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

#data splitting according to the hierarchical structure of hospitals and clients
'''
# no offense but this is useless now
def split_training_data(train_dataset, num_hospitals=3, clients_per_hospital=3):
    
    #level 1: Spliting into Hospitals
    total_len = len(train_dataset)
    hospital_size = total_len // num_hospitals
    hospital_lengths = [hospital_size] * num_hospitals
    hospital_lengths[-1] += total_len - sum(hospital_lengths) #adds remainder to the last hospital if total_len is not perfectly divisible
    
    hospital_datasets = random_split(
        train_dataset, 
        hospital_lengths, 
        generator=torch.Generator().manual_seed(42)
    )
    
    all_hospital_clients = []

    # Level 2: Splitting Hospital Data into Clients
    for hospital_subset in hospital_datasets:
        client_size = len(hospital_subset) // clients_per_hospital
        client_lengths = [client_size] * clients_per_hospital
        client_lengths[-1] += len(hospital_subset) - sum(client_lengths)
        
        client_datasets = random_split(
            hospital_subset, 
            client_lengths,
            generator=torch.Generator().manual_seed(42)
        )
        all_hospital_clients.append(client_datasets)
        
    return all_hospital_clients
'''

# ask anagha
def get_best_ensemble_weights(models, val_loader, device):
    #Grid search to find the best weights (acc to algo5 of paper)
    print("Starting Grid Search for Best Ensemble Weights...")
    
    # Cacheing validation predictions for speed
    print("   Caching validation predictions...")
    all_probs = [[], [], []] # Stores [AlexNet, ResNet, Net] probabilities
    true_labels = []
    
    # Set models to evaluation mode
    for m in models: m.eval()
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            # Altering the label size by converting from [batch, 1] to [batch] if necessary
            if len(labels.shape) > 1: labels = labels.squeeze(1).long()
            else: labels = labels.long()
            
            true_labels.append(labels.cpu())
            
            # Get Softmax Probabilities for each model
            p1 = F.softmax(models[0](inputs), dim=1).cpu()
            p2 = F.softmax(models[1](inputs), dim=1).cpu()
            p3 = F.softmax(models[2](inputs), dim=1).cpu()
            
            all_probs[0].append(p1)
            all_probs[1].append(p2)
            all_probs[2].append(p3)

    # Stack into big tensors for fast math
    P1 = torch.cat(all_probs[0])
    P2 = torch.cat(all_probs[1])
    P3 = torch.cat(all_probs[2])
    Y  = torch.cat(true_labels)
    
    # Grid definition
    # We step by 0.1 
    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Generate all combinations of weights
    combinations = list(itertools.product(steps, repeat=3))
    
    best_acc = 0.0
    best_alpha = (0.33, 0.33, 0.33) # Default equal weights
    
    print(f"   Testing {len(combinations)} weight combinations...")
    
    # Grid search loop
    for alpha in combinations:
        # Normalize weights so they sum to 1.0 
        total = sum(alpha)
        if total == 0: continue # Skip all-zero combination
        
        w1 = alpha[0] / total
        w2 = alpha[1] / total
        w3 = alpha[2] / total
        
        # Weighted Ensemble Probability
        # P_ensemble = w1*P1 + w2*P2 + w3*P3
        P_ensemble = (w1 * P1) + (w2 * P2) + (w3 * P3)
        
        # Get Predictions
        preds = torch.argmax(P_ensemble, dim=1)
        
        # Calculate Accuracy 
        correct = (preds == Y).sum().item()
        accuracy = correct / len(Y)
        
        # Update Best
        if accuracy > best_acc:
            best_acc = accuracy
            best_alpha = (w1, w2, w3)

    print(f"Best Weights Found: {best_alpha}")
    print(f"Validation Accuracy with these weights: {best_acc*100:.2f}%")
    
    return best_alpha

# ask anagha
def evaluate_ensemble(models, weights, test_loader, device):
    #uses the best weights found to test on the final TEST set
    print(f"\nRunning Final Ensemble Test with Weights {weights}...")
    for m in models: m.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if len(labels.shape) > 1: labels = labels.squeeze(1).long()
            else: labels = labels.long()
            
            p1 = F.softmax(models[0](inputs), dim=1)
            p2 = F.softmax(models[1](inputs), dim=1)
            p3 = F.softmax(models[2](inputs), dim=1)
            
            # Weighted Sum
            avg_prob = (weights[0] * p1) + (weights[1] * p2) + (weights[2] * p3)
            
            preds = torch.argmax(avg_prob, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = 100 * correct / total
    return acc

# local function in h_node
def perform_fedavg(client_updates):
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

    total_samples = sum(num_samples for _, num_samples in client_updates)

    avg_state_dict = {}
    first_state_dict = client_updates[0][0] #to know shape

    for key in first_state_dict.keys():
        avg_state_dict[key] = torch.zeros_like(first_state_dict[key]) #same shape filled with zeros

    for state_dict, num_samples in client_updates:
        weight = num_samples / total_samples #how much the client contributes

        for key in state_dict.keys():
            avg_state_dict[key] += state_dict[key] * weight 
    
    torch.save(avg_state_dict, GLOBAL_WEIGHTS_PATH)

    return avg_state_dict #global model weights after averaging


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

# local function for node. this thing just trains
def client_update(global_weights, model_type, client_dataset):
    model = get_model(model_type).to(DEVICE)
    model.load_state_dict(global_weights)

    train_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}")
        avg_loss = total_loss / len(train_loader)

        train_acc = evaluate_model(model, client_dataset)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    return model.state_dict(), len(client_dataset)

 

def evaluate_model(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
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