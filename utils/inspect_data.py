import matplotlib.pyplot as plt
from utils.dataset_loader import get_dataloaders

def inspect():
    train_loader, test_loader, classes = get_dataloaders(batch_size=8)

    print("类别数量:", len(classes))
    print("类别列表:", classes)

    images, labels = next(iter(train_loader))
    print("批次尺寸:", images.shape)

    # 画图检查
    plt.figure(figsize=(10,4))
    for i in range(8):
        img = images[i].permute(1,2,0).numpy()
        img = (img * [0.229,0.224,0.225] + [0.485,0.456,0.406])  # 逆归一化
        plt.subplot(2,4,i+1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis("off")

    plt.show()

if __name__ == "__main__":
    inspect()
