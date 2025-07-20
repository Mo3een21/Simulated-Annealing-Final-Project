import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
import matplotlib.pyplot as plt

class NQueensEnv:
    """Environment for N-Queens problem with state representation"""
    def __init__(self, n):
        self.n = n
        self.board = np.zeros(n, dtype=int)
        self.reset()
    
    def reset(self):
        """Reset board to random configuration"""
        self.board = np.random.randint(0, self.n, size=self.n)
        return self.get_state()
    
    def get_state(self):
        """Convert board to state representation"""
        # State includes: board configuration + conflict counts
        state = []
        
        # Board configuration (one-hot encoded)
        for row in range(self.n):
            row_state = np.zeros(self.n)
            row_state[self.board[row]] = 1
            state.extend(row_state)
        
        # Conflict features
        col_conflicts = np.zeros(self.n)
        diag1_conflicts = np.zeros(2 * self.n - 1)
        diag2_conflicts = np.zeros(2 * self.n - 1)
        
        for row in range(self.n):
            col = self.board[row]
            col_conflicts[col] += 1
            diag1_conflicts[row + col] += 1
            diag2_conflicts[row - col + self.n - 1] += 1
        
        # Normalize conflict counts
        col_conflicts = np.maximum(0, col_conflicts - 1) / self.n
        diag1_conflicts = np.maximum(0, diag1_conflicts - 1) / self.n
        diag2_conflicts = np.maximum(0, diag2_conflicts - 1) / self.n
        
        state.extend(col_conflicts)
        state.extend(diag1_conflicts)
        state.extend(diag2_conflicts)
        
        return np.array(state, dtype=np.float32)
    
    def get_conflicts(self):
        """Calculate total number of conflicts"""
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Same column
                if self.board[i] == self.board[j]:
                    conflicts += 1
                # Same diagonal
                if abs(i - j) == abs(self.board[i] - self.board[j]):
                    conflicts += 1
        return conflicts
    
    def make_move(self, action):
        """Execute action (row, new_col) and return new state, reward"""
        row, new_col = action
        old_col = self.board[row]
        
        if old_col == new_col:
            return self.get_state(), -0.1  # Small penalty for no-op
        
        old_conflicts = self.get_conflicts()
        self.board[row] = new_col
        new_conflicts = self.get_conflicts()
        
        # Reward based on conflict reduction
        reward = old_conflicts - new_conflicts
        
        # Bonus for solving
        if new_conflicts == 0:
            reward += 100
        
        return self.get_state(), reward
    
    def get_all_actions(self):
        """Get all possible actions (row, col) pairs"""
        actions = []
        for row in range(self.n):
            for col in range(self.n):
                if col != self.board[row]:  # Only valid moves
                    actions.append((row, col))
        return actions
    
    def is_solved(self):
        """Check if current state is a solution"""
        return self.get_conflicts() == 0
    
    def copy(self):
        """Create a copy of the environment"""
        new_env = NQueensEnv(self.n)
        new_env.board = self.board.copy()
        return new_env
