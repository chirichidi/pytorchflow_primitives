import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

def get_dataloaders(args):
    train_dataset, val_dataset, test_dataset = get_datasets(args)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def get_transform(args):
    transform = Compose([Resize((args.image_size, args.image_size)), ToTensor()])
    return transform


def get_datasets(args):
    transform = get_transform(args)
    train_val_dataset = MNIST(
        root="../../data", train=True, download=True, transform=transform
    )
    train_dataset, val_dataset = random_split(
        train_val_dataset, [50000, 10000], torch.Generator().manual_seed(42)
    )
    test_dataset = MNIST(
        root="../../data", train=False, download=True, transform=transform
    )

    return train_dataset, val_dataset, test_dataset


def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return optimizer

def get_criteria():
    criteria = torch.nn.CrossEntropyLoss()
    return criteria