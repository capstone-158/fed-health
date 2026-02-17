import torch
import torch.nn as nn

class ImprovedNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedNet, self).__init__()
        
        # Block 1: 3 -> 64 filters
        # Input: 100x100 -> Output: 50x50
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2) 
        )
        
        # Block 2: 64 -> 128 filters
        # Input: 50x50 -> Output: 25x25
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 128 -> 256 filters
        # Input: 25x25 -> Output: 12x12
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4: 256 -> 512 filters (Deep Features)
        # Input: 12x12 -> Output: 6x6
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 5: 512 -> 512 filters (Refinement)
        # Input: 6x6 -> Output: 3x3
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),              # Crucial for preventing overfitting
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        return x