## Plant Disease Classification Based on PlantDoc Dataset

该项目基于 PlantDoc 植物病害数据集，实现从传统机器学习到深度学习（CNN、ResNet50、ViT）的完整植物病害自动识别流程，并包含：

-数据预处理
-特征提取（HOG / SIFT）+ SVM
-CNN 基线模型
-ResNet18 / ResNet50 深度残差网络
-Vision Transformer (ViT-B/16)
-消融实验（学习率、模型深度、数据增强）
-混淆矩阵
-Grad-CAM 可视化
-错误样例分析

## 环境配置
使用 Python 3.9+，推荐创建虚拟环境：
    python -m venv venv
    venv\Scripts\activate  # Windows
主要依赖：
    torch
    torchvision
    numpy
    scikit-learn
    opencv-python
    matplotlib
    tqdm
    pillow
    transformers

## 模型
各模型已经训练好，直接运行测试文件即可（运行前请手动选择测试模型）：
    python cnn/predict_test.py --eval

## 最终结果
方法模型             ||  准确率
MLP                 ||  11.75%
HOG + SVM (RBF)     ||  19.8%
SimpleCNN           ||  13.75%
ResNet18（微调）     ||  57.76%
ResNet50	        ||  61.21%
Vision Transformer	||  66.38%
