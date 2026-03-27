import torch
from .train_function import client_update

def fedavg(client_updates):
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

    return avg_state_dict #global model weights after averaging
