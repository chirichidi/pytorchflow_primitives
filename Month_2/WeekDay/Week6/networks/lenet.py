import torch
import torch.nn as nn

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