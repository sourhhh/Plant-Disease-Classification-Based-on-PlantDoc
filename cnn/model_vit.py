import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit(num_classes):
    # 加载 ImageNet 预训练权重
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # 替换分类头
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
