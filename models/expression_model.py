from torchvision import models
from torch import nn

def get_model(pretrained=True, num_classes=7):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model