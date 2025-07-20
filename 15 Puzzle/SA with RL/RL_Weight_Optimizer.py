import math
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
class ImprovedWeightLearner:
    def __init__(self, regularization=0.01, learning_rate=0.1):
        # Initialize with baseline weights (not all 1s)
        self.weights = np.ones((4, 4))
        # Give more weight to tiles that should be placed first (top-left)
        for i in range(4):
            for j in range(4):
                # Prioritize tiles based on their target position
                self.weights[i, j] = 1.0 + 0.1 * (3 - i) + 0.1 * (3 - j)
        
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.training_data = deque(maxlen=500)
        
        # Track weight history for analysis
        self.weight_history = []
        
    def collect_episode_data(self, env, steps_to_solution):
        """Collect training data from an episode"""
        # Get features from the initial state (not current state)
        if hasattr(env, 'initial_board'):
            # Temporarily swap to initial state to get features
            current_board = env.board.copy()
            current_blank = env.blank_pos
            
            env.board = env.initial_board.copy()
            env.blank_pos = env.initial_board.index(0)
            features = env.get_features()
            
            # Restore current state
            env.board = current_board
            env.blank_pos = current_blank
        else:
            features = env.get_features()
        
        # Store (features, log(1 + steps)) pair
        self.training_data.append((features.flatten(), np.log(1 + steps_to_solution)))
    
    def update_weights(self, n_samples=None):
        """Update weights using gradient descent on the loss function"""
        if len(self.training_data) < 10:
            return
        
        # Use all available data up to n_samples
        if n_samples is None:
            n_samples = len(self.training_data)
        else:
            n_samples = min(n_samples, len(self.training_data))
        
        # Sample from recent runs
        indices = list(range(len(self.training_data)))
        if n_samples < len(self.training_data):
            indices = np.random.choice(len(self.training_data), n_samples, replace=False)
        
        # Compute gradient
        gradient = np.zeros(16)
        total_loss = 0
        
        for idx in indices:
            features, log_steps = self.training_data[idx]
            
            # Current prediction
            phi = np.dot(self.weights.flatten(), features)
            
            # Error
            error = phi - log_steps
            total_loss += error ** 2
            
            # Gradient of squared error
            gradient += 2 * error * features
        
        # Add regularization gradient
        gradient += 2 * self.regularization * self.weights.flatten()
        total_loss += self.regularization * np.sum(self.weights ** 2)
        
        # Normalize gradient
        gradient /= n_samples
        total_loss /= n_samples
        
        # Update weights using gradient descent
        new_weights = self.weights.flatten() - self.learning_rate * gradient
        
        # Ensure weights remain positive and reasonable
        new_weights = np.maximum(new_weights, 0.5)  # Minimum weight
        new_weights = np.minimum(new_weights, 3.0)  # Maximum weight
        
        self.weights = new_weights.reshape(4, 4)
        self.weight_history.append(self.weights.copy())
        
        return total_loss
    
    def compute_loss(self, features, steps):
        """Compute the loss function value"""
        phi = np.sum(self.weights.flatten() * features.flatten())
        target = np.log(1 + steps)
        
        # L(W) = (phi(s) - log(1+t))^2 + lambda*||w||^2
        prediction_loss = (phi - target) ** 2
        regularization_loss = self.regularization * np.sum(self.weights ** 2)
        
        return prediction_loss + regularization_loss

# ==============================================================================
# Enhanced SA Agent with Better Exploration
# ==============================================================================
class AdaptiveSAAgent:
    def __init__(self, env, weight_learner, T0=1.0, cooling_rate=0.999, max_steps=10000):
        self.env = env
        self.weight_learner = weight_learner
        self.T = T0
        self.T0 = T0
        self.cooling_rate = cooling_rate
        self.max_steps = max_steps
        self.steps_taken = 0
        self.trajectory = []
        
        # Adaptive temperature parameters
        self.no_improvement_count = 0
        self.best_phi = env.phi
        
    def reset(self):
        """Reset the agent for a new episode"""
        self.T = self.T0
        self.steps_taken = 0
        self.trajectory = []
        self.no_improvement_count = 0
        self.best_phi = self.env.phi
    
    def step(self, debug=False):
        if self.env.is_solved() or self.steps_taken >= self.max_steps:
            return

        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            return

        # Smart move selection: prefer moves that haven't been tried recently
        if len(self.trajectory) > 5:
            recent_moves = [move[0] for move in self.trajectory[-5:]]
            untried_moves = [m for m in valid_moves if m not in recent_moves]
            if untried_moves:
                valid_moves = untried_moves
        
        tile_pos = random.choice(valid_moves)
        
        old_phi = self.env.phi
        move_info, delta_phi = self.env.make_move(tile_pos)

        # Simulated annealing acceptance
        if delta_phi <= 0:
            accept = True
            if self.env.phi < self.best_phi:
                self.best_phi = self.env.phi
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
        else:
            accept_prob = math.exp(-delta_phi / self.T) if self.T > 0 else 0.0
            accept = random.random() < accept_prob
            self.no_improvement_count += 1

        if accept:
            self.trajectory.append((tile_pos, move_info[1], self.env.phi))
            self.T *= self.cooling_rate
        else:
            self.env.unmake_move(move_info)
            
        # Adaptive temperature reheating
        if self.no_improvement_count > 100:
            self.T = min(self.T * 1.5, self.T0 * 0.5)
            self.no_improvement_count = 0

        self.steps_taken += 1

    def run(self, debug=False):
        """Run until solved or max steps reached"""
        self.reset()
        
        while self.steps_taken < self.max_steps:
            if self.env.is_solved():
                return True, self.steps_taken
            self.step(debug=debug)
            
        return False, self.steps_taken
