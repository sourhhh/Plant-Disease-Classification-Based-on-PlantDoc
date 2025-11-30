import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir="data/train", batch_size=32, val_ratio=0.2):

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    num_total = len(full_dataset)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val

    train_ds, val_ds = random_split(full_dataset, [num_train, num_val])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes
