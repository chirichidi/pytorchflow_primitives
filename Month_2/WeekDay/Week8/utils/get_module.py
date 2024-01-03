import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

from PIL import Image


def get_dataloaders(args):
    transform = get_transform(args)
    train_dataset, val_dataset, test_dataset = get_datasets(transform, args)

    if args.data == "mnist":
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
    elif args.data == "cifar10":
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


def load_image(args):
    image = Image.open(args.target_image_path)
    if args.data == "mnist":
        image = image.convert("L")
    elif args.data == "cifar10":
        image = image.convert("RGB")
    return image


def get_transform(args):
    if args.data == "mnist":
        transform = Compose([Resize((args.image_size, args.image_size)), ToTensor()])
    elif args.data == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        transform = Compose(
            [
                Resize((args.image_size, args.image_size)),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
    else:
        raise NotImplementedError
    return transform


def get_datasets(transform, args):
    if args.data == "mnist":
        from torchvision.datasets import MNIST

        train_val_dataset = MNIST(
            root="../../data", train=True, download=True, transform=transform
        )
        train_dataset, val_dataset = random_split(
            train_val_dataset, [50000, 10000], torch.Generator().manual_seed(42)
        )
        test_dataset = MNIST(
            root="../../data", train=False, download=True, transform=transform
        )
    elif args.data == "cifar10":
        from torchvision.datasets import CIFAR10

        train_val_dataset = CIFAR10(
            root="../../data", train=True, download=True, transform=transform
        )
        train_dataset, val_dataset = random_split(
            train_val_dataset, [40000, 10000], torch.Generator().manual_seed(42)
        )
        test_dataset = CIFAR10(
            root="../../data", train=False, download=True, transform=transform
        )

    return train_dataset, val_dataset, test_dataset


def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return optimizer


def get_criteria():
    criteria = torch.nn.CrossEntropyLoss()
    return criteria


def get_model(args):
    if args.model == "mlp":
        from networks.mlps import myMLP

        model = myMLP(
            image_size=args.image_size,
            hidden_size=args.hidden_size,
            num_class=args.num_class,
        ).to(args.device)
    elif args.model == "lenet":
        from networks.lenet import LeNet

        model = LeNet(num_class=args.num_class).to(args.device)
    elif args.model == "lenet_inj":
        from networks.lenet import LeNet_inj

        model = LeNet_inj(num_class=args.num_class).to(args.device)
    elif args.model == "lenet_multiconv":
        from networks.lenet import LeNet_multiconv

        model = LeNet_multiconv(num_class=args.num_class).to(args.device)
    elif args.model == "lenet_incep":
        from networks.lenet import LeNet_incep

        model = LeNet_incep(num_class=args.num_class).to(args.device)
    elif args.model == "vgg":
        from networks.vgg import VGG

        model = VGG(num_classes=args.num_class, args=args).to(args.device)
    elif args.model == "resnet":
        from networks.resnet import ResNet

        model = ResNet(num_classes=args.num_class, args=args).to(args.device)
    else:
        raise NotImplementedError

    return model
