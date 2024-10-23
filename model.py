import torch
import torch.nn as nn


class ShadowModel(nn.Module):
    def __init__(self, num_classes, is_cifar=True):
        super(ShadowModel, self).__init__()
        if is_cifar:
            # CNN for CIFAR-10
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Output size after pooling for CIFAR-10
        else:
            # CNN for MNIST
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Output size after pooling for MNIST

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
        self.fc1 = nn.Linear(input_size, 64)  # Hidden layer with 64 units
        self.fc2 = nn.Linear(64, 2)  # Output layer (number of classes: 2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)  # Logits output
        return x

    
