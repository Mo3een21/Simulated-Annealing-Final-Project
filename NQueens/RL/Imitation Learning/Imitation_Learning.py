import numpy as np
import math
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import time
from NQueens import NQueensEnv
from SAAgent import SAAgent

class RLCoach:
    """Reinforcement Learning coach for training the potential function weights."""
    
    def __init__(self, board_size: int, num_episodes: int, max_steps: int = 10000,
                 learning_rate: float = 0.001, c: float = 0.02, 
                 momentum: float = 0.9, weight_decay: float = 0.001):
        self.board_size = board_size
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.c = c  # Scaling constant for targets
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize weights
        self.n_features = board_size + 2 * (2 * board_size - 1)
        scale = np.sqrt(2.0 / self.n_features)
        self.weights = np.random.normal(1.0, scale, size=self.n_features)
        self.weights = np.abs(self.weights)  # Ensure positive
        
        # Momentum for SGD
        self.velocity = np.zeros_like(self.weights)
        
        # Metrics
        self.success_count = 0
        self.loss_history = []
        self.steps_history = []
        self.weights_history = []
        self.success_rate_history = []
    
    def compute_loss_and_gradient(self, trajectory: List[Tuple], initial_board: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute loss and gradient for a successful trajectory."""
        T = len(trajectory)
        total_loss = 0.0
        grad_accum = np.zeros(self.n_features)
        
        # Create temporary environment
        temp_env = NQueensEnv(self.board_size)
        temp_env.set_weights_from_vector(self.weights)
        
        # Set initial board state
        temp_env.board = initial_board.copy()
        temp_env.conflicts_col.fill(0)
        temp_env.conflicts_diag_plus.fill(0)
        temp_env.conflicts_diag_minus.fill(0)
        
        for r in range(self.board_size):
            c = temp_env.board[r]
            temp_env.conflicts_col[c] += 1
            temp_env.conflicts_diag_plus[temp_env.diag_plus_index(r, c)] += 1
            temp_env.conflicts_diag_minus[temp_env.diag_minus_index(r, c)] += 1
        
        temp_env.phi = temp_env.compute_phi()
        
        # Process trajectory
        for i, (row, old_col, new_col, phi_recorded) in enumerate(trajectory):
            # Target: we want φ(C^t) ≈ c * (T - t)
            t = T - i - 1  # Steps remaining
            target = self.c * t
            
            # Get features before move
            features = temp_env.get_feature_vector()
            
            # Compute error and gradient
            error = temp_env.phi - target
            total_loss += error ** 2
            grad_accum += 2 * error * features
            
            # Make move
            temp_env.make_move(row, new_col)
        
        avg_loss = total_loss / T
        avg_grad = grad_accum / T
        
        # Add L2 regularization
        avg_grad += self.weight_decay * self.weights
        
        return avg_loss, avg_grad
    
    def update_weights(self, gradient: np.ndarray):
        """Update weights using momentum SGD."""
        # Momentum update
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        self.weights += self.velocity
        
        # Ensure weights stay positive
        self.weights = np.maximum(self.weights, 0.01)
        
        # Optional: normalize weights to prevent explosion
        self.weights = self.weights / np.mean(self.weights)
    
    def run_episode(self, debug: bool = False) -> Dict:
        """Run one training episode."""
        # Create environment with current weights
        env = NQueensEnv(self.board_size)
        env.set_weights_from_vector(self.weights)
        
        initial_board = env.board.copy()
        
        # Run SA agent
        agent = SAAgent(env, T0=0.8, cooling_rate=0.998, max_steps=self.max_steps, adaptive=True)
        success, steps = agent.run(debug)
        
        result = {
            'success': success,
            'steps': steps,
            'trajectory_length': len(agent.trajectory),
            'initial_board': initial_board,
            'loss': None,
            'gradient': None
        }
        
        if success:
            loss, gradient = self.compute_loss_and_gradient(agent.trajectory, initial_board)
            result['loss'] = loss
            result['gradient'] = gradient
            
            self.success_count += 1
            self.loss_history.append(loss)
            self.steps_history.append(steps)
            
            # Update weights
            self.update_weights(gradient)
        
        return result
    
    def train(self, verbose: bool = True, plot_interval: int = 100):
        """Main training loop."""
        print(f"Starting RL training for {self.board_size}-Queens problem")
        print(f"Episodes: {self.num_episodes}, LR: {self.learning_rate}, c: {self.c}")
        print("-" * 60)
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            result = self.run_episode()
            
            # Track metrics
            if (episode + 1) % plot_interval == 0:
                current_success_rate = (self.success_count / (episode + 1)) * 100
                self.success_rate_history.append(current_success_rate)
                self.weights_history.append(self.weights.copy())
                
                if verbose:
                    avg_loss = np.mean(self.loss_history[-plot_interval:]) if self.loss_history else 0
                    avg_steps = np.mean(self.steps_history[-plot_interval:]) if self.steps_history else 0
                    
                    print(f"\nEpisode {episode + 1}/{self.num_episodes}:")
                    print(f"  Success Rate: {current_success_rate:.2f}%")
                    print(f"  Avg Loss (recent): {avg_loss:.6f}")
                    print(f"  Avg Steps (recent): {avg_steps:.1f}")
                    print(f"  Weight Stats - Mean: {np.mean(self.weights):.3f}, "
                          f"Std: {np.std(self.weights):.3f}")
        
        training_time = time.time() - start_time
        
        # Final statistics
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total Time: {training_time:.2f} seconds")
        print(f"Final Success Rate: {(self.success_count / self.num_episodes) * 100:.2f}%")
        if self.loss_history:
            print(f"Final Avg Loss: {np.mean(self.loss_history[-100:]):.6f}")
        if self.steps_history:
            print(f"Final Avg Steps: {np.mean(self.steps_history[-100:]):.1f}")
        
        # Plot results
        self.plot_results()
        
        return self.weights
    
    def plot_results(self):
        """Plot training metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Success rate
        episodes = np.arange(1, len(self.success_rate_history) + 1) * 100
        ax1.plot(episodes, self.success_rate_history, 'b-', linewidth=2)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate During Training')
        ax1.grid(True, alpha=0.3)
        
        # Loss history
        if self.loss_history:
            ax2.plot(self.loss_history, 'r-', alpha=0.5)
            # Moving average
            window = min(100, len(self.loss_history) // 10)
            if window > 1:
                ma = np.convolve(self.loss_history, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(self.loss_history)), ma, 'r-', linewidth=2)
            ax2.set_xlabel('Successful Episodes')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # Steps history
        if self.steps_history:
            ax3.plot(self.steps_history, 'g-', alpha=0.5)
            # Moving average
            window = min(100, len(self.steps_history) // 10)
            if window > 1:
                ma = np.convolve(self.steps_history, np.ones(window)/window, mode='valid')
                ax3.plot(range(window-1, len(self.steps_history)), ma, 'g-', linewidth=2)
            ax3.set_xlabel('Successful Episodes')
            ax3.set_ylabel('Steps to Solution')
            ax3.set_title('Steps Required for Solution')
            ax3.grid(True, alpha=0.3)
        
        # Weight evolution
        if self.weights_history:
            weights_array = np.array(self.weights_history)
            for i in range(min(10, weights_array.shape[1])):  # Plot first 10 weights
                ax4.plot(episodes, weights_array[:, i], alpha=0.7, label=f'w{i}')
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Weight Value')
            ax4.set_title('Weight Evolution (First 10 weights)')
            ax4.legend(ncol=2, fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def test_learned_weights(self, num_tests: int = 100):
        """Test the learned weights on new instances."""
        print(f"\nTesting learned weights on {num_tests} new instances...")
        
        successes = 0
        total_steps = []
        
        for _ in range(num_tests):
            env = NQueensEnv(self.board_size)
            env.set_weights_from_vector(self.weights)
            
            agent = SAAgent(env, T0=0.8, cooling_rate=0.998, max_steps=self.max_steps)
            success, steps = agent.run()
            
            if success:
                successes += 1
                total_steps.append(steps)
        
        success_rate = (successes / num_tests) * 100
        avg_steps = np.mean(total_steps) if total_steps else 0
        
        print(f"Test Success Rate: {success_rate:.2f}%")
        print(f"Average Steps (successful): {avg_steps:.1f}")
        
        return success_rate, avg_steps


def main():
    """Main function to run the RL training."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Training parameters
    config = {
        'board_size': 8,
        'num_episodes': 2000,
        'max_steps': 10000,
        'learning_rate': 0.001,
        'c': 0.02,  # Scaling factor for targets
        'momentum': 0.9,
        'weight_decay': 0.0001
    }
    
    # Create and train coach
    coach = RLCoach(**config)
    learned_weights = coach.train(verbose=True, plot_interval=50)
    
    # Test learned weights
    coach.test_learned_weights(num_tests=100)
    
    # Save learned weights
    np.save(f'learned_weights_n{config["board_size"]}.npy', learned_weights)
    print(f"\nLearned weights saved to 'learned_weights_n{config['board_size']}.npy'")
    
    # Demonstrate a single run with learned weights
    print("\n" + "=" * 60)
    print("Demo: Solving N-Queens with learned weights")
    env = NQueensEnv(config['board_size'])
    env.set_weights_from_vector(learned_weights)
    
    print(f"Initial board (φ = {env.phi:.4f}):")
    print(env)
    
    agent = SAAgent(env, T0=0.8, cooling_rate=0.998, max_steps=3000)
    success, steps = agent.run()
    
    if success:
        print(f"\nSolution found in {steps} steps!")
        print(f"Final board (φ = {env.phi:.4f}):")
        print(env)
    else:
        print(f"\nNo solution found in {steps} steps.")


if __name__ == "__main__":
    main()