import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import random
import time

# Configuration
BATCH_SIZE = 128    
LEARNING_RATE = 0.001 

class EdgeServer:
    def __init__(self, client_id, model_architecture, dataset, device):
        # NOTE: We pass model_architecture just to know the structure, 
        # but we don't store it. We receive the global model in train().
        self.client_id = client_id
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.device = device
        self.num_samples = len(dataset)
        
    def train(self, global_model, epochs):
        # 1. Create a local copy of the global model
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()
        
        # 2. Optimizer with MOMENTUM (Crucial for 'Net' model)
        # Weight decay helps prevent overfitting
        optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        print(f"    Client {self.client_id}: Training {epochs} epochs...", end=" ")
        
        # 3. Local Training Loop
        for epoch in range(epochs):
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Fix label dimensions if necessary
                if len(labels.shape) > 1:
                    labels = labels.squeeze(1).long()
                else:
                    labels = labels.long()
                
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        print("Done.")
        
        # 4. Return the new weights and the number of samples used
        return local_model.state_dict(), self.num_samples

class Hospital:
    def __init__(self, hospital_id, model_architecture, client_datasets, device):
        self.hospital_id = hospital_id
        self.device = device
        self.global_model = model_architecture.to(device)
        self.clients = []
        
        # Initialize Clients (Edge Servers)
        for i, dataset in enumerate(client_datasets):
            # We pass the model architecture so the client knows what to expect
            client = EdgeServer(i, model_architecture, dataset, device)
            self.clients.append(client)
            
    def run_round(self, clients_to_select=2, local_epochs=20):
        """
        Runs one round of Federated Learning:
        1. Select random clients
        2. Train them locally
        3. Aggregate weights (FedAvg)
        """
        # Randomly select subset of clients (e.g., 2 out of 3)
        selected_clients = random.sample(self.clients, min(clients_to_select, len(self.clients)))
        
        collected_weights = []
        total_samples = 0
        
        print(f"  Hospital {self.hospital_id} (Round Start):")
        
        # Distributed Training
        for client in selected_clients:
            w, n = client.train(self.global_model, epochs=local_epochs)
            collected_weights.append((w, n))
            total_samples += n
            
        # FedAvg Aggregation
        # Start with a zeroed-out state dict
        avg_weights = copy.deepcopy(self.global_model.state_dict())
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float)
            
        # Add weighted contributions
        for w, n in collected_weights:
            coeff = n / total_samples
            for key in w.keys():
                avg_weights[key] += w[key] * coeff
        
        # Update Global Model
        self.global_model.load_state_dict(avg_weights)
        return self.global_model

    def evaluate(self, test_dataset):
        """
        Evaluates the global model on the Hospital's test data.
        """
        self.global_model.eval()
        loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if len(labels.shape) > 1: labels = labels.squeeze(1).long()
                else: labels = labels.long()
                
                outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if total == 0: return 0.0
        return 100 * correct / total