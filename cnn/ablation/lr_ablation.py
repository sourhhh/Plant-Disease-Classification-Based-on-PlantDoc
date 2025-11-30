import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model_resnet import build_resnet18
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_loader():
    full = datasets.ImageFolder("data/train", train_tf)
    num_classes = len(full.classes)
    val_size = int(0.2 * len(full))
    train_size = len(full) - val_size
    train_set, val_set = random_split(full, [train_size, val_size])
    val_set.dataset.transform = val_tf
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    return train_loader, val_loader, num_classes

def train_one_epoch(model, loader, c, opt):
    model.train()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = c(out, y)
        loss.backward()
        opt.step()
    # 不返回 loss，因为这里我们只关心最终 val_acc
    return

def eval_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def lr_ablation():
    lrs = [1e-1, 1e-2, 1e-3]
    results = []

    train_loader, val_loader, num_classes = get_loader()

    for lr in lrs:
        print(f"\n===== LR = {lr} =====")
        model = build_resnet18(num_classes).to(device)
        c = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=lr)

        # 简短训练 3 epoch（节省时间，但能体现趋势）
        for _ in range(3):
            train_one_epoch(model, train_loader, c, opt)

        acc = eval_model(model, val_loader)
        results.append(acc)
        print(f"Val Acc = {acc:.4f}")

    # 绘制折线图
    plt.plot(lrs, results, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy")
    plt.title("Ablation Study - Learning Rate")
    plt.grid()
    plt.savefig("cnn/ablation/lr_ablation.png")
    plt.show()

if __name__ == "__main__":
    lr_ablation()
