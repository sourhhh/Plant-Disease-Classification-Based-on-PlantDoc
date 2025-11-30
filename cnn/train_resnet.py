import torch
import torch.nn as nn
import torch.optim as optim
from model_resnet import build_resnet18
from utils_cnn import get_dataloaders
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_resnet(epochs=20, batch_size=32, lr=1e-4):

    print("使用设备:", device)

    train_loader, val_loader, classes = get_dataloaders(batch_size=batch_size)

    model = build_resnet18(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 预训练模型常用的：学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0

    for epoch in range(epochs):
        # -------------------- TRAIN --------------------
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # -------------------- VAL --------------------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"TrainLoss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"ValLoss={val_loss:.4f} Acc={val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "cnn/best_resnet18.pth")
            print(">>> 保存最佳模型")

        scheduler.step()

    # -------------------- Plot Curves --------------------
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.savefig("cnn/resnet18_loss.png")
    plt.close()

    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.savefig("cnn/resnet18_acc.png")
    plt.close()

    print("训练完成，最佳模型保存在 cnn/best_resnet18.pth")


if __name__ == "__main__":
    train_resnet()
