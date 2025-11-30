import argparse
import os
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model_resnet import build_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 1. 参数解析
# ==========================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_aug", type=int, default=1, help="1=使用数据增强, 0=不使用数据增强")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


# ==========================
# 2. 训练代码
# ==========================
def train_model(use_aug=1, batch_size=32, lr=1e-3, epochs=5):

    # 数据增强策略
    if use_aug:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载原始 train 集
    dataset = datasets.ImageFolder("data/train", transform=train_transform)

    # 自动划分 val 集
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    val_set.dataset.transform = test_transform  # 验证集不使用增强

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # 加载模型
    model = build_resnet18(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_acc_list = []
    val_acc_list = []

    print("\n==============================")
    print(" 开始训练 (增强={} )".format(use_aug))
    print("==============================\n")

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_list.append(train_acc)

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_acc_list.append(val_acc)

        print("Epoch [{}/{}]  Train Acc = {:.4f} | Val Acc = {:.4f}".format(
            epoch+1, epochs, train_acc, val_acc
        ))

    # 保存曲线图
    out_path = "cnn/ablation"
    os.makedirs(out_path, exist_ok=True)

    tag = "with_aug" if use_aug else "no_aug"

    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve ({})".format(tag))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    save_path = f"{out_path}/aug_{tag}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"图像已保存：{save_path}")
    print("最终 Val Acc =", val_acc_list[-1])

    return val_acc_list[-1]


if __name__ == "__main__":
    args = get_args()
    train_model(use_aug=args.use_aug, batch_size=args.batch_size,
                lr=args.lr, epochs=args.epochs)
