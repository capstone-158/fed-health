import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import models
from src.resnet_model import AdaptedResNet18
from src.alexnet_model import AdaptedAlexNet
from src.net_model import ImprovedNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
BATCH_SIZE = 128 #can use 64 initially for debugging 
LOCAL_EPOCHS = 20 #can use 2 initially for debugging
NUM_CLASSES = 9

def get_model(model_type):
    if model_type == "resnet":
        return AdaptedResNet18(NUM_CLASSES)
    elif model_type == "alexnet":
        return AdaptedAlexNet(NUM_CLASSES)
    elif model_type == "net":
        return ImprovedNet(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def client_update(global_weights, model_type, client_dataset):
    model = get_model(model_type).to(DEVICE)

    # Load global weights, same for all clients so all start from same point
    model.load_state_dict(global_weights)

    train_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(LOCAL_EPOCHS):
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #print(f"[Client] Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    return model.state_dict(), len(client_dataset)