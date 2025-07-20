import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
# ==============================================================================
# Deep Q-Network for Learning Optimal Moves
# ==============================================================================
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=12):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)