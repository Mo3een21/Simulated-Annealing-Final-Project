import numpy as np
import math
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import time
from NQueens import NQueensEnv

class SAAgent:
    """Simulated Annealing agent with adaptive temperature scheduling."""
    
    def __init__(self, env: NQueensEnv, T0: float = 1.0, cooling_rate: float = 0.999, 
                 max_steps: int = 10000, adaptive: bool = True):
        self.env = env
        self.T0 = T0
        self.T = T0
        self.cooling_rate = cooling_rate
        self.max_steps = max_steps
        self.steps_taken = 0
        self.trajectory = []
        self.adaptive = adaptive
        self.accept_history = []  # Track acceptance rate
        
    def step(self, debug: bool = False) -> bool:
        """Perform one SA step. Returns True if move was accepted."""
        n = self.env.n
        
        # Select random move
        row = random.randint(0, n - 1)
        current_col = self.env.board[row]
        new_col = current_col
        while new_col == current_col:
            new_col = random.randint(0, n - 1)
        
        # Try move
        move_result = self.env.make_move(row, new_col)
        if move_result is None:
            return False
        
        move_info, delta_phi = move_result
        
        # Acceptance decision
        if delta_phi <= 0:
            accept = True
        else:
            accept_prob = math.exp(-delta_phi / self.T) if self.T > 0 else 0.0
            accept = random.random() < accept_prob
        
        if accept:
            # Record state after accepting move
            self.trajectory.append((row, current_col, new_col, self.env.phi))
            self.accept_history.append(1)
            
            # Aggressive cooling on acceptance
            self.T *= self.cooling_rate ** 10
        else:
            # Undo move
            self.env.unmake_move(move_info)
            self.accept_history.append(0)
            
            # Reheat on rejection
            self.T /= self.cooling_rate
        
        # Adaptive temperature control
        if self.adaptive and len(self.accept_history) >= 100:
            recent_accept_rate = np.mean(self.accept_history[-100:])
            if recent_accept_rate < 0.2:  # Too cold
                self.T *= 1.1
            elif recent_accept_rate > 0.8:  # Too hot
                self.T *= 0.9
        
        self.steps_taken += 1
        
        if debug:
            print(f"Step {self.steps_taken}, T={self.T:.4f}, φ={self.env.phi:.4f}, "
                  f"Δφ={delta_phi:.4f}, Accept={accept}")
        
        return accept
    
    def run(self, debug: bool = False) -> Tuple[bool, int]:
        """Run SA until solution found or max steps reached."""
        while self.steps_taken < self.max_steps:
            if self.env.is_solution():
                return True, self.steps_taken
            self.step(debug)
        return False, self.steps_taken
