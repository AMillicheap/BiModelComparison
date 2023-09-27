"""
BMH-22814 Research Assistant Interview Task - models
Alex Millicheap
"""

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, data_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(data_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class LogisticRegressionModel(nn.Module):
    def __init__(self, data_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(data_size, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
