import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ShadowModel(nn.Module):
    def __init__(self, num_classes):
        super(ShadowModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# class ShadowModel(nn.Module):
#     def __init__(self, num_classes, is_cifar=True):
#         super(ShadowModel, self).__init__()
#         if is_cifar:
#             # for CIFAR-10
#             self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
#             self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#             self.pool = nn.MaxPool2d(2, 2)
#             self.fc1 = nn.Linear(64 * 8 * 8, 128)
#         else:
#             # for MNIST
#             self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
#             self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#             self.pool = nn.MaxPool2d(2, 2)
#             self.fc1 = nn.Linear(64 * 7 * 7, 128)

#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         if len(x.shape) == 3:  # MNIST input (1 channel)
#             x = x.unsqueeze(1)  # Add channel dimension for MNIST
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         # x = x.view(-1, self.fc1.in_features)
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

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
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x