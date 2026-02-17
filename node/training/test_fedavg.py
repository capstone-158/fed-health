import torch
from src.alexnet_model import AdaptedAlexNet
from src.resnet_model import AdaptedResNet18
from src.net_model import ImprovedNet # Using the new ImprovedNet
from src.dataset import get_colon_data, split_training_data
from src.federated import Hospital
import os

# Training settings as in paper
COMMS_ROUNDS = 20       
LOCAL_EPOCHS = 20     
NUM_CLIENTS = 3         
CLIENTS_ACTIVE = 2      
NUM_CLASSES = 9         

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple MPS used")
    else:
        device = torch.device("cpu")
        print("MPS not available")
    
    # 1. Load Data
    train_d, val_d, test_d = get_colon_data()
    
    # 2. Hierarchical Split
    client_allocations = split_training_data(train_d, num_hospitals=3, clients_per_hospital=NUM_CLIENTS)
    
    # 3. Initialize Hospitals
    print("Initializing Hospitals...")
    h1 = Hospital(1, AdaptedAlexNet(NUM_CLASSES), client_allocations[0], device)
    h2 = Hospital(2, AdaptedResNet18(NUM_CLASSES), client_allocations[1], device)
    h3 = Hospital(3, ImprovedNet(NUM_CLASSES), client_allocations[2], device)
    
    hospitals = [h1, h2, h3]
    model_names = ["AlexNet", "ResNet18", "Net"]
    
    # 4. Run Training
    print(f"\nStarting Training for {COMMS_ROUNDS} Rounds...")
    
    for round_idx in range(COMMS_ROUNDS):
        print(f"\n--- Round {round_idx + 1} ---")
        for i, h in enumerate(hospitals):
            h.run_round(clients_to_select=CLIENTS_ACTIVE, local_epochs=LOCAL_EPOCHS)
            
        if (round_idx + 1) % 5 == 0:
            print(f"Evaluating at Round {round_idx+1}...")
            for i, h in enumerate(hospitals):
                acc = h.evaluate(test_d)
                print(f"  H{i+1} ({model_names[i]}) Test Accuracy: {acc:.2f}%")

    # Saving the models
    print("\nSaving Final Global Models...")
    try:
        torch.save(h1.global_model.state_dict(), "alexnet_global.pth")
        torch.save(h2.global_model.state_dict(), "resnet_global.pth")
        torch.save(h3.global_model.state_dict(), "net_global.pth")
        print(f"Models saved successfully in: {os.getcwd()}")
    except Exception as e:
        print(f"Error saving models: {e}")
    # -------------------------

    # 5. Final Evaluation
    print("\n--- Final FedAvg Benchmark Results ---")
    targets = [84.70, 90.40, 82.40]
    print(f"{'Model':<15} | {'Target':<15} | {'Achieved':<15}")
    for i, h in enumerate(hospitals):
        final_acc = h.evaluate(test_d)
        print(f"{model_names[i]:<15} | {targets[i]:<15} | {final_acc:.2f}%")

if __name__ == "__main__":
    main()