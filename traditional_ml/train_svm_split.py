import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from extract_features import extract_hog_feature
from utils_ml import load_image_paths
from tqdm import tqdm
import joblib
import os

def prepare_train_val(path):
    img_paths, classes = load_image_paths(path)
    data, labels = [], []

    print(f"读取训练集：{len(img_paths)} 张图片")

    for p, label in tqdm(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        feat = extract_hog_feature(img)
        data.append(feat)
        labels.append(label)

    X = np.array(data)
    y = np.array(labels)

    # Train/Val 划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"训练集：{len(X_train)}   验证集：{len(X_val)}")

    return X_train, X_val, y_train, y_val, classes


def prepare_test(path):
    img_paths, _ = load_image_paths(path)
    data = []

    print(f"读取测试集：{len(img_paths)} 张图片")

    for p in tqdm(img_paths):
        img = cv2.imread(p)
        feat = extract_hog_feature(img)
        data.append(feat)

    return np.array(data), img_paths


if __name__ == "__main__":
    # Step 1: 加载 train/val
    X_train, X_val, y_train, y_val, classes = prepare_train_val("data/train")

    # Step 2: 训练 SVM
    print("训练 SVM 中...")
    clf = SVC(kernel="rbf", C=10, gamma=0.001)
    clf.fit(X_train, y_train)

    # Step 3: 验证集评估
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\n📌 验证集准确率：{val_acc:.4f}")

    # Step 4: 测试集预测
    X_test, test_paths = prepare_test("data/test")
    test_preds = clf.predict(X_test)

    # Step 5: 保存预测结果
    with open("traditional_ml/svm_test_predictions.txt", "w") as f:
        for p, pred in zip(test_paths, test_preds):
            f.write(f"{os.path.basename(p)}, {classes[pred]}\n")

    # Step 6: 保存模型
    joblib.dump(clf, "traditional_ml/svm_hog_model.pkl")
    print("\n模型已保存：svm_hog_model.pkl")
