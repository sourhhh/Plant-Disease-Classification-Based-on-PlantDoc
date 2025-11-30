import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 你的模型
from model_resnet import build_resnet18
from model_resnet50 import build_resnet50
from model_vit import build_vit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===========================================
# 选择模型类型
# ===========================================
MODEL_TYPE = "resnet18"    # 可选："resnet18", "resnet50", "vit"

MODEL_PATH = {
    "resnet18": "cnn/best_resnet18.pth",
    "resnet50": "cnn/best_resnet50.pth",
    "vit":      "cnn/best_vit.pth"
}[MODEL_TYPE]


# ===========================================
# 数据预处理（与验证集一致）
# ===========================================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ===========================================
# 从 test_0 读取真实标签
# ===========================================
TEST_DIR = "data/test_0"

def load_test_data():
    dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader, dataset.classes


# ===========================================
# 根据模型类型加载模型
# ===========================================
def load_model(num_classes):
    if MODEL_TYPE == "resnet18":
        model = build_resnet18(num_classes)
    elif MODEL_TYPE == "resnet50":
        model = build_resnet50(num_classes)
    elif MODEL_TYPE == "vit":
        model = build_vit(num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


# ===========================================
# 计算混淆矩阵
# ===========================================
def compute_confusion_matrix():
    loader, classes = load_test_data()
    num_classes = len(classes)

    model = load_model(num_classes)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm, classes


# ===========================================
# 绘制混淆矩阵
# ===========================================
def plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {MODEL_TYPE}")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"

    # 写数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    save_path = f"cnn/confusion_matrix_{MODEL_TYPE}.png"
    plt.savefig(save_path)
    print(f"\n混淆矩阵已保存为：{save_path}")
    plt.show()


# ===========================================
# 主函数
# ===========================================
def main():
    print("生成混淆矩阵中...")
    cm, classes = compute_confusion_matrix()
    plot_confusion_matrix(cm, classes, normalize=True)


if __name__ == "__main__":
    main()
