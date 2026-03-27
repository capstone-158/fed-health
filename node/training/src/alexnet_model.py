import torch
import torch.nn as nn
from torchvision import models

class AdaptedAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AdaptedAlexNet, self).__init__()

        # Load pretrained AlexNet
        try:
            from torchvision.models import AlexNet_Weights
            self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        except ImportError:
            self.model = models.alexnet(pretrained=True)

        # Adjust for 100x100 input
        self.model.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Replace final classifier layer
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)