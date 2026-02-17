import torch
import torch.nn as nn
from torchvision import models

class AdaptedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(AdaptedResNet18, self).__init__()
        #loading pretrained weights
        try:
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except ImportError:
            self.model = models.resnet18(pretrained=True)
        
        #downsizing to 100x100 
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)