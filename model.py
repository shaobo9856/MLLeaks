import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ImprovedShadowModel(nn.Module):
    def __init__(self, num_classes, is_cifar=True):
        super(ImprovedShadowModel, self).__init__()
        channel = 3 if is_cifar else 1
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class ShadowModel(nn.Module):
    def __init__(self, num_classes, is_cifar=True):
        super(ShadowModel, self).__init__()
        if is_cifar:
            # for CIFAR-10
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
        else:
            # for MNIST
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if len(x.shape) == 3:  # MNIST input (1 channel)
            x = x.unsqueeze(1)  # Add channel dimension for MNIST
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # x = x.view(-1, self.fc1.in_features)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AttackModel(nn.Module):
    def __init__(self, input_size):
        super(AttackModel, self).__init__()
        self.hidden = nn.Linear(input_size, 64) 
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = nn.functional.relu(self.hidden(x))
        x = self.output(x)  # Logits output
        return nn.functional.softmax(x,dim=1)

class ImprovedAttackModel(nn.Module):
    def __init__(self, input_size):
        super(ImprovedAttackModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x