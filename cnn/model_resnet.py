import torch.nn as nn
from torchvision.models import resnet18

def build_resnet18(num_classes=27):
    model = resnet18(weights="IMAGENET1K_V1")  # 预训练
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # 替换分类层
    return model
