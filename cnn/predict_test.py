import os
import torch
from PIL import Image
from torchvision import transforms
from utils_cnn import get_dataloaders
from model_resnet import build_resnet18
from model_resnet50 import build_resnet50
from model_vit import build_vit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPE = "vit"
MODEL_PATH = {
    "resnet18": "cnn/best_resnet18.pth",
    "resnet50": "cnn/best_resnet50.pth",
    "vit":      "cnn/best_vit.pth"
}[MODEL_TYPE]

TEST_DIR = "data/test"
OUTPUT_CSV = f"cnn/submission_{MODEL_TYPE}.csv"

# 加载类名（与训练一致）
_, _, classes = get_dataloaders(batch_size=1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


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


def predict_single(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)

    return classes[pred.item()]


def main():
    model = load_model(len(classes))

    images = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    print(f"发现 {len(images)} 张无标签测试图片，开始预测...")

    results = []

    for img_name in images:
        img_path = os.path.join(TEST_DIR, img_name)
        pred_label = predict_single(model, img_path)
        results.append((img_name, pred_label))
        print(f"{img_name} → {pred_label}")

    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("image,prediction\n")
        for name, label in results:
            f.write(f"{name},{label}\n")

    print(f"\n预测结束，CSV 已生成：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()

