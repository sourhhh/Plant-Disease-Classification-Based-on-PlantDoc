# DataLoader载入器
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .transforms import get_train_transform, get_test_transform

def get_dataloaders(train_dir="data/train", test_dir="data/test", batch_size=32):
    train_ds = ImageFolder(train_dir, transform=get_train_transform())
    test_ds  = ImageFolder(test_dir, transform=get_test_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ds.classes
