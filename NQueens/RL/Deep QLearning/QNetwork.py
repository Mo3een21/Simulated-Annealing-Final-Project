import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    """Neural network for Q-value approximation"""
    def __init__(self, state_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + 2, hidden_size)  # +2 for action (row, col)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)  # Output Q-value
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state, action):
        """Forward pass: takes state and action, returns Q-value"""
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        q_value = self.fc4(x)
        
        return q_value