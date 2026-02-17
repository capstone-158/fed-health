import torch
import os
from torch.utils.data import DataLoader
from src.dataset import get_colon_data
from src.ensemble import get_best_ensemble_weights, evaluate_ensemble

# Import Model Architectures
from src.alexnet_model import AdaptedAlexNet
from src.resnet_model import AdaptedResNet18
from src.net_model import ImprovedNet

# Configuration
NUM_CLASSES = 9
BATCH_SIZE = 128

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS acceleration")
    else:
        device = torch.device("cpu")
        
    print("Loading Pre-trained Models")
    
    # Initialize Empty Models
    # We create the skeleton first, then load the weights
    model_alex = AdaptedAlexNet(NUM_CLASSES).to(device)
    model_res  = AdaptedResNet18(NUM_CLASSES).to(device)
    model_net  = ImprovedNet(NUM_CLASSES).to(device)
    
    # Load the Weights (Check if files exist first)
    files = ["alexnet_global.pth", "resnet_global.pth", "net_global.pth"]
    for f in files:
        if not os.path.exists(f):
            print(f"ERROR: Could not find '{f}'")
            return

    try:
        model_alex.load_state_dict(torch.load("alexnet_global.pth", map_location=device))
        model_res.load_state_dict(torch.load("resnet_global.pth", map_location=device))
        model_net.load_state_dict(torch.load("net_global.pth", map_location=device))
        print("All weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    models = [model_alex, model_res, model_net]

    #Load Data
    print("Loading Data")
    _, val_d, test_d = get_colon_data()
    
    val_loader = DataLoader(val_d, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    
    # Run Algorithm (Grid Search)
    print("\nWeight tuning")
    # This finds the best combination (e.g., 0.2, 0.5, 0.3) using Validation Data
    best_weights = get_best_ensemble_weights(models, val_loader, device)
    
    #Final Evaluation
    print("\nFinal Evaluation on TEST Set")
    # This applies those weights to the unseen Test Data
    final_acc = evaluate_ensemble(models, best_weights, test_loader, device)
    
    print(f"\nFinal Ensemble Accuracy: {final_acc:.2f}%")
    
    # Baseline Comparison
    print("\n--- Benchmarks ---")
    print(f"ResNet (Individual Best): ~90.4%")
    print(f"AlexNet (Individual):     ~84.7%")
    print(f"Net (Individual):         ~82.4%")
    
    if final_acc > 90.4:
        print(f"Beats the individual models by +{final_acc - 90.4:.2f}%")

if __name__ == "__main__":
    main()