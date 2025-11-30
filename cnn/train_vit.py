import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model_vit import build_vit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================================
# 只有轻微数据增强，避免 ViT 不稳定
# =====================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =====================================
# 自动划分 train / val
# =====================================
def get_loaders(batch_size=8, val_ratio=0.2):
    full_dataset = datasets.ImageFolder("data/train", transform=train_transform)
    num_classes = len(full_dataset.classes)

    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)   # 固定随机种子，保证结果可复现
    )

    # 验证集必须使用 val_transform（不能用训练增强）
    val_dataset.dataset.transform = val_transform

    print(f"训练集数量：{train_size}")
    print(f"验证集数量：{val_size}")
    print(f"类别数量：{num_classes} -> {full_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes


# =====================================
# 训练 1 epoch
# =====================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


# =====================================
# 验证
# =====================================
def validate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


# =====================================
# 主训练函数（自动生成 best_vit.pth）
# =====================================
def train_vit(epochs=20, batch_size=8, lr=3e-5):
    train_loader, val_loader, classes = get_loaders(batch_size)
    num_classes = len(classes)

    model = build_vit(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    print("\n===== 开始训练 ViT =====")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "cnn/best_vit.pth")
            print(f"🔥 保存新最优模型：Val Acc={best_acc:.4f}")

    print("\n训练结束")
    print(f"最佳验证准确率：{best_acc:.4f}")


if __name__ == "__main__":
    train_vit()
