import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader



def get_transforms():
    trin_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5, 0.5),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), inplace=False)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5, 0.5)
    ])

    return trin_transform, test_transform


def get_dataloaders(batch_size:int, data_root:str):
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform,
        download=True)
    
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=test_transform,
        download=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader



    





