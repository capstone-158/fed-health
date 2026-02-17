import torch
import torch.nn.functional as F
import itertools

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
