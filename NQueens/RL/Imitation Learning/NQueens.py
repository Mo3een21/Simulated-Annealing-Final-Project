import numpy as np
import math
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import time

class NQueensEnv:
    """N-Queens environment with learnable weighted potential function."""
    
    def __init__(self, n: int, weights_col=None, weights_diag_plus=None, weights_diag_minus=None):
        self.n = n
        self.board = np.zeros(n, dtype=int)
        
        # Conflict counters
        self.conflicts_col = np.zeros(n, dtype=int)
        self.conflicts_diag_plus = np.zeros(2*n - 1, dtype=int)
        self.conflicts_diag_minus = np.zeros(2*n - 1, dtype=int)
        
        # Initialize weights (learnable parameters)
        if weights_col is None:
            # Xavier/He initialization for better convergence
            scale = np.sqrt(2.0 / (n + (2*n - 1) * 2))
            self.weights_col = np.random.normal(1.0, scale, size=n)
            self.weights_diag_plus = np.random.normal(1.0, scale, size=2*n - 1)
            self.weights_diag_minus = np.random.normal(1.0, scale, size=2*n - 1)
        else:
            self.weights_col = weights_col.copy()
            self.weights_diag_plus = weights_diag_plus.copy()
            self.weights_diag_minus = weights_diag_minus.copy()
        
        # Ensure weights are positive
        self.weights_col = np.abs(self.weights_col)
        self.weights_diag_plus = np.abs(self.weights_diag_plus)
        self.weights_diag_minus = np.abs(self.weights_diag_minus)
        
        self.phi = 0.0
        self.init_board()
    
    def diag_plus_index(self, row: int, col: int) -> int:
        return row + col
    
    def diag_minus_index(self, row: int, col: int) -> int:
        return row - col + (self.n - 1)
    
    def init_board(self):
        """Initialize board with random queen placement."""
        self.board = np.random.randint(0, self.n, size=self.n)
        self.conflicts_col.fill(0)
        self.conflicts_diag_plus.fill(0)
        self.conflicts_diag_minus.fill(0)
        
        for r in range(self.n):
            c = self.board[r]
            self.conflicts_col[c] += 1
            self.conflicts_diag_plus[self.diag_plus_index(r, c)] += 1
            self.conflicts_diag_minus[self.diag_minus_index(r, c)] += 1
        
        self.phi = self.compute_phi()
    
    def compute_phi(self) -> float:
        """Compute potential function: Σ w_i * max(0, c_i - 1)"""
        phi = 0.0
        
        # Columns
        for j in range(self.n):
            phi += self.weights_col[j] * max(0, self.conflicts_col[j] - 1)
        
        # Diagonals
        for idx in range(2*self.n - 1):
            phi += self.weights_diag_plus[idx] * max(0, self.conflicts_diag_plus[idx] - 1)
            phi += self.weights_diag_minus[idx] * max(0, self.conflicts_diag_minus[idx] - 1)
        
        return phi
    
    def update_line(self, conflicts_array: np.ndarray, weights_array: np.ndarray, 
                   index: int, delta: int) -> float:
        """Update conflict count and return change in phi."""
        old_count = conflicts_array[index]
        old_feature = max(0, old_count - 1)
        new_count = old_count + delta
        new_feature = max(0, new_count - 1)
        conflicts_array[index] = new_count
        return weights_array[index] * (new_feature - old_feature)
    
    def make_move(self, row: int, new_col: int) -> Optional[Tuple]:
        """Move queen and update phi incrementally."""
        old_col = self.board[row]
        if old_col == new_col:
            return None
        
        delta_phi = 0.0
        
        # Remove from old position
        delta_phi += self.update_line(self.conflicts_col, self.weights_col, old_col, -1)
        dp_idx_old = self.diag_plus_index(row, old_col)
        delta_phi += self.update_line(self.conflicts_diag_plus, self.weights_diag_plus, dp_idx_old, -1)
        dm_idx_old = self.diag_minus_index(row, old_col)
        delta_phi += self.update_line(self.conflicts_diag_minus, self.weights_diag_minus, dm_idx_old, -1)
        
        # Update board
        self.board[row] = new_col
        
        # Add to new position
        delta_phi += self.update_line(self.conflicts_col, self.weights_col, new_col, +1)
        dp_idx_new = self.diag_plus_index(row, new_col)
        delta_phi += self.update_line(self.conflicts_diag_plus, self.weights_diag_plus, dp_idx_new, +1)
        dm_idx_new = self.diag_minus_index(row, new_col)
        delta_phi += self.update_line(self.conflicts_diag_minus, self.weights_diag_minus, dm_idx_new, +1)
        
        self.phi += delta_phi
        
        move_info = (row, old_col, new_col, dp_idx_old, dp_idx_new, dm_idx_old, dm_idx_new)
        return move_info, delta_phi
    
    def unmake_move(self, move_info: Tuple) -> float:
        """Undo a move."""
        row, old_col, new_col, dp_idx_old, dp_idx_new, dm_idx_old, dm_idx_new = move_info
        delta_phi = 0.0
        
        # Remove from new position
        delta_phi += self.update_line(self.conflicts_col, self.weights_col, new_col, -1)
        delta_phi += self.update_line(self.conflicts_diag_plus, self.weights_diag_plus, dp_idx_new, -1)
        delta_phi += self.update_line(self.conflicts_diag_minus, self.weights_diag_minus, dm_idx_new, -1)
        
        # Restore to old position
        self.board[row] = old_col
        delta_phi += self.update_line(self.conflicts_col, self.weights_col, old_col, +1)
        delta_phi += self.update_line(self.conflicts_diag_plus, self.weights_diag_plus, dp_idx_old, +1)
        delta_phi += self.update_line(self.conflicts_diag_minus, self.weights_diag_minus, dm_idx_old, +1)
        
        self.phi += delta_phi
        return delta_phi
    
    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector: [max(0, c_i - 1) for all lines]"""
        f_cols = np.maximum(0, self.conflicts_col - 1)
        f_diag_plus = np.maximum(0, self.conflicts_diag_plus - 1)
        f_diag_minus = np.maximum(0, self.conflicts_diag_minus - 1)
        return np.concatenate([f_cols, f_diag_plus, f_diag_minus])
    
    def get_all_weights(self) -> np.ndarray:
        """Get all weights as a single vector."""
        return np.concatenate([self.weights_col, self.weights_diag_plus, self.weights_diag_minus])
    
    def set_weights_from_vector(self, weights: np.ndarray):
        """Set weights from a single vector."""
        n_cols = self.n
        n_diags = 2 * self.n - 1
        self.weights_col = weights[:n_cols].copy()
        self.weights_diag_plus = weights[n_cols:n_cols+n_diags].copy()
        self.weights_diag_minus = weights[n_cols+n_diags:].copy()
        
        # Ensure weights are positive
        self.weights_col = np.abs(self.weights_col)
        self.weights_diag_plus = np.abs(self.weights_diag_plus)
        self.weights_diag_minus = np.abs(self.weights_diag_minus)
        
        # Recompute phi with new weights
        self.phi = self.compute_phi()
    
    def is_solution(self) -> bool:
        """Check if current state is a solution."""
        return self.phi < 0.001
    
    def count_conflicts(self) -> int:
        """Count total number of conflicts."""
        conflicts = 0
        conflicts += np.sum(np.maximum(0, self.conflicts_col - 1))
        conflicts += np.sum(np.maximum(0, self.conflicts_diag_plus - 1))
        conflicts += np.sum(np.maximum(0, self.conflicts_diag_minus - 1))
        return int(conflicts)
    
    def __str__(self) -> str:
        s = ""
        for r in range(self.n):
            row_str = ""
            for c in range(self.n):
                if self.board[r] == c:
                    row_str += "Q "
                else:
                    row_str += ". "
            s += row_str.rstrip() + "\n"
        return s