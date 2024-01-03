import torch
import torch.nn as nn


class VGGConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0

        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.module(x)


class VGG_A(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()

        self.vgg_block1 = nn.Sequential(
            VGGConv(in_channels=3, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block2 = nn.Sequential(
            VGGConv(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block3 = nn.Sequential(
            VGGConv(in_channels=128, out_channels=256),
            VGGConv(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block4 = nn.Sequential(
            VGGConv(in_channels=256, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block5 = nn.Sequential(
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        in_features = 512 if args.image_size == 32 else 25088
        self.vgg_classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):  # x : [batch_size, height, width, channel]
        batch_size = x.shape[0]
        x = self.vgg_block1(x)
        x = self.vgg_block2(x)
        x = self.vgg_block3(x)
        x = self.vgg_block4(x)
        x = self.vgg_block5(x)  # x : [batch_size, 7, 7, 512]
        x = x.reshape(batch_size, -1)  # x : [batch_size, 7 * 7 * 512]
        x = self.vgg_classifier(x)
        return x


class VGG_B(VGG_A):
    def __init__(self, num_classes, args):
        super().__init__(num_classes, args)
        self.vgg_block1 = nn.Sequential(
            VGGConv(in_channels=3, out_channels=64),
            VGGConv(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block2 = nn.Sequential(
            VGGConv(in_channels=64, out_channels=128),
            VGGConv(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class VGG_C(VGG_B):
    def __init__(self, num_classes, args):
        super().__init__(num_classes, args)
        self.vgg_block3 = nn.Sequential(
            VGGConv(in_channels=128, out_channels=256),
            VGGConv(in_channels=256, out_channels=256),
            VGGConv(in_channels=256, out_channels=256, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block4 = nn.Sequential(
            VGGConv(in_channels=256, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block5 = nn.Sequential(
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class VGG_D(VGG_B):
    def __init__(self, num_classes, args):
        super().__init__(num_classes, args)
        self.vgg_block3 = nn.Sequential(
            VGGConv(in_channels=128, out_channels=256),
            VGGConv(in_channels=256, out_channels=256),
            VGGConv(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block4 = nn.Sequential(
            VGGConv(in_channels=256, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block5 = nn.Sequential(
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class VGG_E(VGG_D):
    def __init__(self, num_classes, args):
        super().__init__(num_classes, args)
        self.vgg_block3 = nn.Sequential(
            VGGConv(in_channels=128, out_channels=256),
            VGGConv(in_channels=256, out_channels=256),
            VGGConv(in_channels=256, out_channels=256),
            VGGConv(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block4 = nn.Sequential(
            VGGConv(in_channels=256, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.vgg_block5 = nn.Sequential(
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            VGGConv(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class VGG(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()

        if args.vgg_conf == "a":
            self.model = VGG_A(num_classes, args)
        if args.vgg_conf == "b":
            self.model = VGG_B(num_classes, args)
        if args.vgg_conf == "c":
            self.model = VGG_C(num_classes, args)
        if args.vgg_conf == "d":
            self.model = VGG_D(num_classes, args)
        if args.vgg_conf == "e":
            self.model = VGG_E(num_classes, args)

    def forward(self, x):
        return self.model(x)
