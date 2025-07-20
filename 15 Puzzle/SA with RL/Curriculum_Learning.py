import math
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from SA_with_HC import SAAgent
from RL_Weight_Optimizer import ImprovedWeightLearner, AdaptiveSAAgent
from Puzzle15 import Puzzle15Env
# ==============================================================================
# Curriculum Learning Manager
# ==============================================================================
class SmartCurriculumManager:
    def __init__(self, start_difficulty=5, max_difficulty=100, increment=2):
        self.current_difficulty = start_difficulty
        self.max_difficulty = max_difficulty
        self.increment = increment
        self.success_threshold = 0.75  # Lower threshold
        self.recent_successes = deque(maxlen=30)
        self.stuck_counter = 0
        
    def get_difficulty(self):
        """Get current difficulty level (number of shuffle steps)"""
        return self.current_difficulty
    
    def update(self, success):
        """Update difficulty based on recent performance"""
        self.recent_successes.append(1 if success else 0)
        
        if len(self.recent_successes) >= 20:
            success_rate = sum(self.recent_successes) / len(self.recent_successes)
            
            # Increase difficulty if doing well
            if success_rate >= self.success_threshold:
                old_difficulty = self.current_difficulty
                self.current_difficulty = min(self.current_difficulty + self.increment, 
                                            self.max_difficulty)
                if self.current_difficulty > old_difficulty:
                    print(f"Advancing to difficulty {self.current_difficulty} (success rate: {success_rate:.2f})")
                    self.recent_successes.clear()
                    self.stuck_counter = 0
            else:
                self.stuck_counter += 1
                
                # If stuck for too long, temporarily reduce difficulty
                if self.stuck_counter > 50 and self.current_difficulty > 10:
                    self.current_difficulty = max(5, self.current_difficulty - 5)
                    print(f"Temporarily reducing difficulty to {self.current_difficulty}")
                    self.stuck_counter = 0
                    self.recent_successes.clear()

# ==============================================================================
# Main Training Function
# ==============================================================================
def train_with_adaptive_curriculum(n_episodes=1000, n_initial_runs=30):
    """Train the weight learner using adaptive curriculum learning"""
    
    # Initialize components
    weight_learner = ImprovedWeightLearner(regularization=0.01, learning_rate=0.05)
    curriculum = SmartCurriculumManager(start_difficulty=5, max_difficulty=100)
    
    # Statistics tracking
    success_history = []
    steps_history = []
    difficulty_history = []
    loss_history = []
    
    print("Phase 1: Initial baseline collection")
    print("="*60)
    
    # Phase 1: Collect initial data with mixed difficulties
    for i in range(n_initial_runs):
        # Use varied difficulties in initial phase
        difficulty = random.choice([5, 7, 10, 12, 15])
        env = Puzzle15Env(shuffle_steps=difficulty, use_learned_weights=False)
        agent = AdaptiveSAAgent(env, weight_learner, T0=1.0, cooling_rate=0.999)
        
        solved, steps = agent.run()
        
        if solved:
            weight_learner.collect_episode_data(env, steps)
            
        success_history.append(1 if solved else 0)
        steps_history.append(steps if solved else agent.max_steps)
        difficulty_history.append(difficulty)
        
        if (i + 1) % 10 == 0:
            recent_success = np.mean(success_history[-10:])
            print(f"Initial phase {i+1}/{n_initial_runs}: Success rate={recent_success:.2f}")
    
    # Initial weight update
    print("\nPhase 2: Training with learned weights")
    print("="*60)
    
    loss = weight_learner.update_weights()
    if loss is not None:
        print(f"Initial loss: {loss:.4f}")
    print(f"Initial weights range: [{weight_learner.weights.min():.2f}, {weight_learner.weights.max():.2f}]")
    
    # Phase 2: Continue training with learned weights
    for episode in range(n_initial_runs, n_episodes):
        difficulty = curriculum.get_difficulty()
        
        # Create environment with learned weights
        env = Puzzle15Env(shuffle_steps=difficulty, use_learned_weights=True)
        env.set_weights(weight_learner.weights)
        
        # Store initial state for feature extraction
        env.initial_board = env.board.copy()
        
        agent = AdaptiveSAAgent(env, weight_learner, T0=1.0, cooling_rate=0.999)
        
        solved, steps = agent.run()
        
        # Collect data and update
        if solved:
            weight_learner.collect_episode_data(env, steps)
        
        # Update curriculum
        curriculum.update(solved)
        
        # Update weights more frequently when learning
        if episode < 200 and episode % 10 == 0:
            loss = weight_learner.update_weights()
            if loss is not None:
                loss_history.append(loss)
        elif episode % 20 == 0:
            loss = weight_learner.update_weights()
            if loss is not None:
                loss_history.append(loss)
        
        # Track statistics
        success_history.append(1 if solved else 0)
        steps_history.append(steps if solved else agent.max_steps)
        difficulty_history.append(difficulty)
        
        # Progress report
        if episode % 50 == 0:
            recent_success = np.mean(success_history[-50:])
            recent_steps = [s for s, succ in zip(steps_history[-50:], success_history[-50:]) if succ]
            avg_steps = np.mean(recent_steps) if recent_steps else 0
            
            print(f"\nEpisode {episode}:")
            print(f"  Difficulty: {difficulty} shuffle steps")
            print(f"  Success Rate: {recent_success:.2f}")
            print(f"  Avg Steps (successful): {avg_steps:.1f}")
            print(f"  Weights: min={weight_learner.weights.min():.2f}, "
                  f"max={weight_learner.weights.max():.2f}, "
                  f"mean={weight_learner.weights.mean():.2f}")
            if loss_history:
                print(f"  Recent loss: {loss_history[-1]:.4f}")
    
    return weight_learner, success_history, steps_history, difficulty_history, loss_history

# ==============================================================================
# Better Evaluation Function
# ==============================================================================
def evaluate_comprehensive(weight_learner, test_cases=30):
    """Comprehensive evaluation of learned vs standard heuristic"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)
    
    difficulties = [10, 20, 30, 50, 70]
    
    for difficulty in difficulties:
        print(f"\nTesting on {difficulty} shuffle steps ({test_cases} test cases):")
        
        # Test with learned weights
        learned_results = {'solved': 0, 'steps': [], 'timeouts': 0}
        
        for _ in range(test_cases):
            env = Puzzle15Env(shuffle_steps=difficulty, use_learned_weights=True)
            env.set_weights(weight_learner.weights)
            env.initial_board = env.board.copy()
            
            agent = AdaptiveSAAgent(env, weight_learner, T0=1.0, cooling_rate=0.999, max_steps=20000)
            
            solved, steps = agent.run()
            if solved:
                learned_results['solved'] += 1
                learned_results['steps'].append(steps)
            else:
                learned_results['timeouts'] += 1
        
        # Test with standard heuristic
        standard_results = {'solved': 0, 'steps': [], 'timeouts': 0}
        
        for _ in range(test_cases):
            env = Puzzle15Env(shuffle_steps=difficulty, use_learned_weights=False)
            agent = AdaptiveSAAgent(env, weight_learner, T0=1.0, cooling_rate=0.999, max_steps=20000)
            
            solved, steps = agent.run()
            if solved:
                standard_results['solved'] += 1
                standard_results['steps'].append(steps)
            else:
                standard_results['timeouts'] += 1
        
        # Report results
        print(f"  Learned weights:")
        print(f"    Solved: {learned_results['solved']}/{test_cases} ({learned_results['solved']/test_cases*100:.1f}%)")
        if learned_results['steps']:
            print(f"    Avg steps: {np.mean(learned_results['steps']):.0f} (median: {np.median(learned_results['steps']):.0f})")
        
        print(f"  Standard heuristic:")
        print(f"    Solved: {standard_results['solved']}/{test_cases} ({standard_results['solved']/test_cases*100:.1f}%)")
        if standard_results['steps']:
            print(f"    Avg steps: {np.mean(standard_results['steps']):.0f} (median: {np.median(standard_results['steps']):.0f})")
    
    # Display final learned weights
    print("\nLearned Weight Matrix:")
    print("(Position [i,j] = weight for tile that belongs at position [i,j])")
    for i in range(4):
        print("  " + " ".join(f"{weight_learner.weights[i,j]:4.2f}" for j in range(4)))
# ==============================================================================
# Plotting Function
# ==============================================================================
def plot_training_progress(success_history, steps_history, difficulty_history, weights_history):
    """Plot training progress"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Success rate over time
    window = 50
    success_smooth = [np.mean(success_history[max(0, i-window):i+1]) 
                      for i in range(len(success_history))]
    axes[0, 0].plot(success_smooth)
    axes[0, 0].set_title('Success Rate (smoothed)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].grid(True)
    
    # Steps to solution
    successful_episodes = [(i, s) for i, (s, succ) in enumerate(zip(steps_history, success_history)) if succ]
    if successful_episodes:
        episodes, steps = zip(*successful_episodes)
        axes[0, 1].scatter(episodes, steps, alpha=0.5, s=10)
        axes[0, 1].set_title('Steps to Solution (successful episodes)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
    
    # Difficulty progression
    axes[1, 0].plot(difficulty_history)
    axes[1, 0].set_title('Curriculum Difficulty')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Shuffle Steps')
    axes[1, 0].grid(True)
    
    # Weight evolution
    if weights_history:
        # Plot evolution of a few selected weights
        weights_array = np.array(weights_history)
        n_weights_to_plot = min(5, weights_array.shape[1])
        
        for i in range(n_weights_to_plot):
            weight_idx = i * (16 // n_weights_to_plot)
            row, col = weight_idx // 4, weight_idx % 4
            axes[1, 1].plot(weights_array[:, row, col], 
                           label=f'w[{row},{col}]')
        
        axes[1, 1].set_title('Weight Evolution')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Weight Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('15puzzle_training_progress.png')
    plt.show()
# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("Advanced 15-Puzzle Solver with Reinforcement Learning")
    print("="*60)
    
    # Train the system
    weight_learner, success_hist, steps_hist, diff_hist, loss_hist = \
        train_with_adaptive_curriculum(n_episodes=600, n_initial_runs=30)
    
    # Evaluate performance
    evaluate_comprehensive(weight_learner, test_cases=20)
    
    # Plot results
    plot_training_progress(success_hist, steps_hist, diff_hist, weight_learner.weight_history)
