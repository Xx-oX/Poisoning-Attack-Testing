# Lenet-5 Model architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) # 1 input channel, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120) # 16 channels, 5x5 kernel
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x)) # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # no activation function
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features