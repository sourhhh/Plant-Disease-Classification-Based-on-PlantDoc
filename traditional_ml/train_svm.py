import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from extract_features import extract_hog_feature
from utils_ml import load_image_paths
from tqdm import tqdm
import joblib

def prepare_dataset(path, use_hog=True):
    data = []
    labels = []

    img_paths, classes = load_image_paths(path)

    print(f"读取数据：{len(img_paths)} 张图像")

    for p, label in tqdm(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue

        if use_hog:
            feat = extract_hog_feature(img)
        else:
            raise NotImplementedError("目前只支持 HOG 特征")

        data.append(feat)
        labels.append(label)

    return np.array(data), np.array(labels), classes


if __name__ == "__main__":
    print("正在提取训练集特征...")
    X_train, y_train, classes = prepare_dataset("data/train")

    print("训练 SVM 中...")
    clf = SVC(kernel='rbf', C=10, gamma=0.001)
    clf.fit(X_train, y_train)

    print("训练完成，开始测试集评估...")
    X_test, y_test, _ = prepare_dataset("data/test")

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"测试集准确率：{acc:.4f}")

    joblib.dump(clf, "traditional_ml/svm_hog_model.pkl")
    print("模型已保存：svm_hog_model.pkl")
