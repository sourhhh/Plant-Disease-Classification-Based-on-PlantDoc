import cv2
import numpy as np
from skimage.feature import hog

# 统一输入图像大小（建议 256×256）
IMG_SIZE = (256, 256)

def extract_hog_feature(image):
    """
    输入：RGB 或 BGR 图像
    输出：统一长度的 HOG 特征向量
    """

    # Step 1: 统一 resize，防止特征维度不一致
    image = cv2.resize(image, IMG_SIZE)

    # Step 2: 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: 提取 HOG
    feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )

    return feature

def extract_sift_feature(image):
    """
    提取 SIFT 特征，并进行平均池化得到固定维度（128）
    """
    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kps, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        return np.zeros(128)

    return descriptors.mean(axis=0)