import torch
import torch.nn as nn
import torch.optim as optim
from model_cnn import BetterCNN
from utils_cnn import get_dataloaders
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_cnn(epochs=15, batch_size=32, lr=1e-3):

    # 加载数据
    train_loader, val_loader, classes = get_dataloaders(batch_size=batch_size)

    # 模型初始化
    model = BetterCNN(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []


    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        # ---------------- TRAIN ----------------
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---------------- VAL ----------------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} "
              f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "cnn/best_cnn.pth")
            best_val_acc = val_acc
            print(">>> 保存最佳模型")

    # ------- 绘制 Loss 曲线 -------
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("cnn/loss_curve.png")
    plt.close()

    # ------- 绘制 Accuracy 曲线 -------
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("cnn/acc_curve.png")
    plt.close()

    print("训练完成，模型已保存为 best_cnn.pth")


if __name__ == "__main__":
    train_cnn()
