import os
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TEST_GT_DIR = "data/test_0"     # 含真实标签的测试集
PRED_CSV = "cnn/submission_vit.csv"   # 第一步生成的预测CSV

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_ground_truth():
    """
    读取 data/test_0 的真实标签（使用 ImageFolder 自动读取）
    返回 dict: { filename : true_label }
    """
    dataset = datasets.ImageFolder(TEST_GT_DIR, transform=transform)
    samples = dataset.samples  # (path, class_idx)
    classes = dataset.classes

    gt = {}

    for path, class_idx in samples:
        filename = os.path.basename(path)
        gt[filename] = classes[class_idx]

    print(f"真实标签读取完毕，共 {len(gt)} 张图片")
    return gt


def load_predictions():
    """
    从 CSV 中读取模型预测结果
    返回 dict: { filename : predicted_label }
    """
    pred = {}
    with open(PRED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred[row["image"]] = row["prediction"]

    print(f"读取到 {len(pred)} 条预测记录")
    return pred


def evaluate():
    gt = load_ground_truth()
    pred = load_predictions()

    correct = 0
    total = 0

    for filename, true_label in gt.items():
        if filename in pred:
            if pred[filename] == true_label:
                correct += 1
            total += 1

    acc = correct / total
    print(f"\n模型在 test_0 上的准确率：{acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    evaluate()
