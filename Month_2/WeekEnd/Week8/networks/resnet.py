import torch
import torch.nn as nn

LAYER18 = [2, 2, 2, 2]
LAYER34 = [3, 4, 6, 3]
LAYER50 = [3, 4, 6, 3]
LAYER101 = [3, 4, 23, 3]
LAYER152 = [3, 8, 36, 3]

CHANNEL33 = [64, 128, 256, 512]
CHANNEL131 = [256, 512, 1024, 2048]


class InputPart(nn.Module):
    def __init__(self, input_channel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(
        self, x
    ):  # x : [batch_size, channel(3), height(224, 32), width(224, 32)]
        x = self.conv(x)  # half size
        x = self.pool(x)  # half size
        return x


class OutputPart(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = 512 if config in [18, 34] else 2048
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(
        self, x
    ):  # x : [batch_size, channel(512 // 2048), height(7, 1), width(7, 1)]
        batch_size = x.shape[0]
        x = self.pool(x)  # x : [batch_size, channel(512 // 2048), 1, 1]
        x = torch.reshape(x, (batch_size, -1))
        x = self.fc(x)
        return x


class Conv(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride, padding, has_relu=True
    ):
        super().__init__()
        if has_relu:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, first_block=False):
        super().__init__()
        self.first_block = first_block
        stride = 1
        if self.first_block:
            stride = 2
            self.size_matching = Conv(
                in_channel, out_channel, 3, stride, 1, has_relu=False
            )

        self.conv1 = Conv(in_channel, out_channel, 3, stride, 1)
        self.conv2 = Conv(out_channel, out_channel, 3, 1, 1, has_relu=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        copy_x = x.clone()
        if self.first_block:
            copy_x = self.size_matching(copy_x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + copy_x
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, first_block=False):
        super().__init__()

        middle_channel = out_channel // 4
        stride = 2 if first_block else 1

        self.conv1 = Conv(in_channel, middle_channel, 1, stride, 0)
        self.conv2 = Conv(middle_channel, middle_channel, 3, 1, 1)
        self.conv3 = Conv(middle_channel, out_channel, 1, 1, 0, has_relu=False)
        self.relu = nn.ReLU()

        self.size_matching = Conv(in_channel, out_channel, 3, stride, 1, has_relu=False)

    def forward(self, x):
        copy_x = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.size_matching(copy_x)
        x = self.relu(x)
        return x


class MiddlPart(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config == 18:
            num_layer = LAYER18
            num_channel = CHANNEL33
            self.TARGET = Block
        elif config == 34:
            num_layer = LAYER34
            num_channel = CHANNEL33
            self.TARGET = Block
        elif config == 50:
            num_layer = LAYER50
            num_channel = CHANNEL131
            self.TARGET = BottleNeck
        elif config == 101:
            num_layer = LAYER101
            num_channel = CHANNEL131
            self.TARGET = BottleNeck
        elif config == 152:
            num_layer = LAYER152
            num_channel = CHANNEL131
            self.TARGET = BottleNeck

        self.Layer1 = self._make_layer(num_layer[0], 64, num_channel[0])
        self.Layer2 = self._make_layer(
            num_layer[1], num_channel[0], num_channel[1], first_block=True
        )
        self.Layer3 = self._make_layer(
            num_layer[2], num_channel[1], num_channel[2], first_block=True
        )
        self.Layer4 = self._make_layer(
            num_layer[3], num_channel[2], num_channel[3], first_block=True
        )

    def _make_layer(self, num_layer, in_channel, out_channel, first_block=False):
        layer = [self.TARGET(in_channel, out_channel, first_block=first_block)]  # 특수 처리
        for i in range(num_layer - 1):
            layer.append(self.TARGET(out_channel, out_channel))  # 리스트

        return nn.ModuleList(layer)
        # return nn.Sequential(*layer)

    def forward(self, x):
        for module in self.Layer1:
            x = module(x)
        for module in self.Layer2:
            x = module(x)
        for module in self.Layer3:
            x = module(x)
        for module in self.Layer4:
            x = module(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        # args안에 정보를 보니 타겟하는 config를 알아내고
        self.config = args.resnet_config
        # 그 config를 기준으로
        # 입력부 생성
        self.input_part = InputPart()
        # 중간부 생성
        self.middle_part = MiddlPart(self.config)
        # 출력부 생성
        self.output_part = OutputPart(self.config, num_classes)

    def forward(self, x):
        x = self.input_part(x)
        x = self.middle_part(x)
        x = self.output_part(x)
        return x
