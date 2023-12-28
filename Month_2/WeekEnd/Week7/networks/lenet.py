import torch
import torch.nn as nn



class LeNet_incep(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
        ) 
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(in_features=84, out_features=num_class)

    def forward(self, x):
        batch_size = x.shape[0]

        x1 = self.conv1_1(x) # (batch_size, 6, 32, 32)
        x2 = self.conv1_2(x) # (batch_size, 6, 32, 32)
        x3 = self.conv1_3(x) # (batch_size, 6, 32, 32)

        x = torch.cat((x1, x2, x3), dim=1) # (batch_size, 18, 32, 32)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = torch.reshape(x, (batch_size, 400))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LeNet_multiconv(nn.Module):
    def __init__(self, num_class, num_conv1 = 4, num_conv2 = 3):
        super().__init__()
        self.num_class = num_class

        conv1 = []
        conv_in_channels = 3
        conv_in_padding = 2
        for i in range(num_conv1):
            conv_in_channels = 3 if i == 0 else 6
            conv_in_padding = 0 if i == (num_conv1 - 1) else 2
            module = nn.Sequential(
                nn.Conv2d(
                    in_channels=conv_in_channels,
                    out_channels=6,
                    kernel_size=5,
                    stride=1,
                    padding=conv_in_padding,
                ),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU()
            )
            conv1.append(module)
        self.conv1 = nn.ModuleList(conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv2 = []
        conv_in_channels = 6
        conv_in_padding = 0
        for i in range(3):
            conv_in_channels = 6 if i == 0 else 16
            conv_in_padding = 0 if i == (num_conv2 - 1) else 2
            module = nn.Sequential(
                nn.Conv2d(
                    in_channels=conv_in_channels,
                    out_channels=16,
                    kernel_size=5,
                    stride=1,
                    padding=conv_in_padding,
                ),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU()
            )
            conv2.append(module)
        self.conv2 = nn.ModuleList(conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(in_features=84, out_features=num_class)

    def forward(self, x):
        batch_size = x.shape[0]

        for conv in self.conv1:
            x = conv(x)
        x = self.pool1(x)

        for conv in self.conv2:
            x = conv(x)
        x = self.pool2(x)

        x = torch.reshape(x, (batch_size, 400))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    

class LeNet_inj(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inj_Linear1 = nn.Sequential(
            nn.Linear(in_features=1176, out_features=2048),
            nn.ReLU(),
        )
        self.inj_Linear2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1176),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(in_features=84, out_features=num_class)

    def forward(self, x): # x : (batch_size, 3, 32, 32)
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.pool1(x)  # (batch_size, 6, 14, 14)

        _, c, h, w = x.shape
        x = torch.reshape(x, (batch_size, -1)) # (batch_size, 1176)
        x = self.inj_Linear1(x) # (batch_size, 2048)
        x = self.inj_Linear2(x) # (batch_size, 1176)
        x = torch.reshape(x, (batch_size, c, h, w)) # (batch_size, 6, 14, 14)

        x = self.conv2(x)
        x = self.pool2(x)

        _, c, h, w = x.shape
        x = torch.reshape(x, (batch_size, c * h * w))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    

class LeNet(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(in_features=84, out_features=num_class)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = torch.reshape(x, (batch_size, 400))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x