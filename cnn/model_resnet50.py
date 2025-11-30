import torch.nn as nn
from torchvision.models import resnet50

def build_resnet50(num_classes=27):
    model = resnet50(weights="IMAGENET1K_V2")   # 预训练权重
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
