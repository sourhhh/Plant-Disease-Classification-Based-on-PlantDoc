import os
import csv
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 导入你的模型
from model_resnet import build_resnet18
from model_resnet50 import build_resnet50
from model_vit import build_vit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# 模型类型选择
# =============================
MODEL_TYPE = "resnet18"

MODEL_PATH = {
    "resnet18": "cnn/best_resnet18.pth",
    "resnet50": "cnn/best_resnet50.pth",
    "vit":      "cnn/best_vit.pth"
}[MODEL_TYPE]

# 测试集路径（带标签）
TEST_DIR = "data/test_0"

# 错误样例保存路径
SAVE_DIR = "cnn/error_cases"
os.makedirs(SAVE_DIR, exist_ok=True)

# CSV 文件路径
CSV_PATH = os.path.join(SAVE_DIR, "error_cases.csv")

# =============================
# 数据预处理
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =============================
# 加载测试集
# =============================
def load_test_data():
    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader, dataset.classes, dataset.samples


# =============================
# 加载模型
# =============================
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


# =============================
# 保存一张错误样本图
# =============================
def save_error_image(img_path, true_label, pred_label, idx):
    img = Image.open(img_path).convert("RGB")

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)

    save_img_path = os.path.join(SAVE_DIR, f"error_{idx}.png")
    plt.savefig(save_img_path, bbox_inches='tight')
    plt.close()

    return save_img_path


# =============================
# 主流程：查找错误并保存 CSV
# =============================
def visualize_error_cases(max_errors=200):
    loader, classes, samples = load_test_data()
    model = load_model(len(classes))

    errors_found = 0
    csv_rows = []

    print("\n开始查找模型预测错误的样本...\n")

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            if preds.item() != labels.item():  # 错误预测
                img_path, _ = samples[i]
                true_label = classes[labels.item()]
                pred_label = classes[preds.item()]

                print(f"[ERROR] {os.path.basename(img_path)}: True={true_label}, Pred={pred_label}")

                # 保存图像
                save_img_path = save_error_image(img_path, true_label, pred_label, errors_found)

                # 保存 CSV 记录
                csv_rows.append([save_img_path, true_label, pred_label])

                errors_found += 1
                if errors_found >= max_errors:
                    break

    # =============================
    # 写入 CSV 文件
    # =============================
    with open(CSV_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "pred_label"])
        writer.writerows(csv_rows)

    print(f"\n共找到 {errors_found} 个错误案例。")
    print(f"错误案例图像已保存到：{SAVE_DIR}")
    print(f"CSV 文件已保存为：{CSV_PATH}")


if __name__ == "__main__":
    visualize_error_cases(max_errors=200)
