import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import os

def get_colon_data():
    data_path = 'processed_colon.pt' 
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f" '{data_path}' not found!")

    #loading pre-processed data 100x100
    print(f"Loading pre-processed data from {data_path}...")
    data = torch.load(data_path)
    
    # Creating TensorDatasets
    train_dataset = TensorDataset(data['train_x'], data['train_y'])
    val_dataset = TensorDataset(data['val_x'], data['val_y'])
    test_dataset = TensorDataset(data['test_x'], data['test_y'])
    
    return train_dataset, val_dataset, test_dataset

#data splitting according to the hierarchical structure of hospitals and clients
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