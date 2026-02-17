import torch
from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import TensorDataset

def preprocess_and_save():
    print("Resizing 28x28 -> 100x100)...")
    
    # Define the resize transform
    resize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100, 100), antialias=True),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Load the raw data (Train, Val, Test)
    print("   Loading Raw Data...")
    train_raw = PathMNIST(split='train', transform=resize_transform, download=True)
    val_raw = PathMNIST(split='val', transform=resize_transform, download=True)
    test_raw = PathMNIST(split='test', transform=resize_transform, download=True)

    # Helper to process a dataset into a single Tensor
    def process_split(dataset, name):
        print(f"   Processing {name} set ({len(dataset)} images)...")
        loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
        all_x = []
        all_y = []
        
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)
            
        # Concatenate into one big tensor
        x_tensor = torch.cat(all_x)
        y_tensor = torch.cat(all_y)
        return x_tensor, y_tensor

    # Process all splits
    train_x, train_y = process_split(train_raw, "Train")
    val_x, val_y = process_split(val_raw, "Val")
    test_x, test_y = process_split(test_raw, "Test")

    # Save to disk
    print("Saving to 'processed_colon.pt'...")
    torch.save({
        'train_x': train_x, 'train_y': train_y,
        'val_x': val_x, 'val_y': val_y,
        'test_x': test_x, 'test_y': test_y
    }, 'processed_colon.pt')
    
    print("Resized data saved to 'processed_colon.pt'")

if __name__ == "__main__":
    preprocess_and_save()