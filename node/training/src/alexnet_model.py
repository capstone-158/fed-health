import torch
import torch.nn as nn
from torchvision import models

class AdaptedAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AdaptedAlexNet, self).__init__()

        #loading pretrained alexnet weights
        try:
            from torchvision.models import AlexNet_Weights
            original_alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        except ImportError:
            original_alexnet = models.alexnet(pretrained=True)
        
        self.features = original_alexnet.features
        
        # input size 100x100 -> adaptive pooling to 6x6 output
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = original_alexnet.classifier
        
        #number of classes in dataset(9)
        num_ftrs = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x