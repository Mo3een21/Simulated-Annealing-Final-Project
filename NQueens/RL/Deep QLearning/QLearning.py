import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
import matplotlib.pyplot as plt
from QNetwork import QNetwork
from NQueens import NQueensEnv

class ReplayBuffer:
    """Experience replay buffer for storing transitions"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        """Add transition to buffer"""
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """Sample batch of transitions"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """Q-Learning agent with neural network"""
    def __init__(self, n, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.n = n
        self.state_size = n * n + n + 2 * (2 * n - 1)  # Board + conflicts
        
        # Q-network and target network
        self.q_network = QNetwork(self.state_size)
        self.target_network = QNetwork(self.state_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = ReplayBuffer()
        
        # For tracking history
        self.episode_history = []
        
    def get_action(self, env, state, temperature=0.8, use_epsilon_greedy=True):
        """Select action using epsilon-greedy or Boltzmann exploration"""
        actions = env.get_all_actions()
        
        if use_epsilon_greedy and random.random() < self.epsilon:
            # Random action
            return random.choice(actions)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Calculate Q-values for all actions
        q_values = []
        action_tensors = []
        
        for action in actions:
            action_tensor = torch.FloatTensor([action[0] / self.n, action[1] / self.n])
            action_tensors.append(action_tensor)
            
            with torch.no_grad():
                q_value = self.q_network(state_tensor, action_tensor.unsqueeze(0))
                q_values.append(q_value.item())
        
        q_values = np.array(q_values)
        
        if temperature > 0:
            # Boltzmann exploration (for SA-style selection)
            # Use negative Q-values as energy (higher Q = lower energy = better)
            energies = -q_values / temperature
            probabilities = np.exp(energies - np.max(energies))  # Numerical stability
            probabilities /= probabilities.sum()
            
            action_idx = np.random.choice(len(actions), p=probabilities)
        else:
            # Greedy selection
            action_idx = np.argmax(q_values)
        
        return actions[action_idx]
    
    def train_step(self, batch_size=32):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.FloatTensor([[t[1][0] / self.n, t[1][1] / self.n] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        # Current Q-values
        current_q_values = self.q_network(states, actions).squeeze()
        
        # Next Q-values from target network
        next_q_values = torch.zeros(batch_size)
        
        for i in range(batch_size):
            if not dones[i]:
                env_copy = NQueensEnv(self.n)
                env_copy.board = np.zeros(self.n, dtype=int)  # Temporary
                next_actions = env_copy.get_all_actions()
                
                max_q = float('-inf')
                for next_action in next_actions:
                    action_tensor = torch.FloatTensor([next_action[0] / self.n, next_action[1] / self.n]).unsqueeze(0)
                    with torch.no_grad():
                        q_val = self.target_network(next_states[i].unsqueeze(0), action_tensor).item()
                        max_q = max(max_q, q_val)
                
                next_q_values[i] = max_q
        
        # Target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def run_sa_episode(self, env, T0=1.0, cooling_rate=0.995, max_steps=5000):
        """Run SA episode using Q-network for action selection"""
        env.reset()
        state = env.get_state()
        temperature = T0
        
        history = []
        step = 0
        
        while step < max_steps and not env.is_solved():
            # Get action using Q-network with SA temperature
            action = self.get_action(env, state, temperature, use_epsilon_greedy=False)
            
            # Store current state
            old_state = state.copy()
            old_board = env.board.copy()
            
            # Make move
            new_state, reward = env.make_move(action)
            
            # SA acceptance criteria
            if reward >= 0:  # Accept improving moves
                accept = True
            else:
                # Accept worsening moves with probability
                accept_prob = math.exp(reward / temperature) if temperature > 0 else 0
                accept = random.random() < accept_prob
            
            if accept:
                # Store transition
                done = env.is_solved()
                self.replay_buffer.push((old_state, action, reward, new_state, done))
                history.append((old_state, action, new_state))
                state = new_state
                temperature *= cooling_rate
            else:
                # Reject move
                env.board = old_board
            
            step += 1
        
        success = env.is_solved()
        self.episode_history = history
        
        return success, step, history


def train_agent(n=8, num_episodes=1000, batch_size=32):
    """Train the Q-learning agent"""
    env = NQueensEnv(n)
    agent = QLearningAgent(n, learning_rate=0.001, gamma=0.95, epsilon=0.1)
    
    success_rate_history = []
    steps_history = []
    
    for episode in range(num_episodes):
        # Run SA episode with Q-network
        success, steps, history = agent.run_sa_episode(env)
        
        if success:
            steps_history.append(steps)
        
        # Train Q-network
        if len(agent.replay_buffer) > batch_size:
            for _ in range(10):  # Multiple training steps per episode
                agent.train_step(batch_size)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        # Track progress
        if episode % 100 == 0:
            recent_successes = sum(1 for s in steps_history[-100:] if s > 0)
            success_rate = recent_successes if episode < 100 else recent_successes / 100
            avg_steps = np.mean(steps_history[-100:]) if steps_history else 0
            
            success_rate_history.append(success_rate)
            
            print(f"Episode {episode}: Success rate: {success_rate:.2%}, "
                  f"Avg steps: {avg_steps:.1f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, success_rate_history, steps_history


def visualize_results(success_rate_history, steps_history):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate
    ax1.plot(range(0, len(success_rate_history) * 100, 100), success_rate_history)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate over Training')
    ax1.grid(True)
    
    # Steps to solution
    if steps_history:
        window = 100
        steps_smooth = [np.mean(steps_history[max(0, i-window):i+1]) 
                       for i in range(len(steps_history))]
        ax2.plot(steps_smooth)
        ax2.set_xlabel('Successful Episode')
        ax2.set_ylabel('Steps to Solution')
        ax2.set_title('Steps to Solution (Moving Average)')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_trained_agent(agent, n=8, num_tests=100):
    """Test the trained agent"""
    env = NQueensEnv(n)
    successes = 0
    total_steps = []
    
    # Disable exploration for testing
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    for _ in range(num_tests):
        success, steps, _ = agent.run_sa_episode(env, T0=0.8, max_steps=20000)
        if success:
            successes += 1
            total_steps.append(steps)
    
    agent.epsilon = original_epsilon
    
    success_rate = successes / num_tests
    avg_steps = np.mean(total_steps) if total_steps else 0
    
    print(f"\nTest Results:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Steps to Solution: {avg_steps:.1f}")
    
    return success_rate, avg_steps


# Main execution
if __name__ == "__main__":
    print("Training Q-Network Agent for N-Queens...")
    
    # Train agent
    agent, success_history, steps_history = train_agent(n=8, num_episodes=10)
    
    # Visualize results
    visualize_results(success_history, steps_history)
    
    # Test trained agent
    test_trained_agent(agent, n=8, num_tests=100)
    
    # Demonstrate a single episode
    print("\nDemonstrating a single episode:")
    env = NQueensEnv(8)
    success, steps, history = agent.run_sa_episode(env, T0=0.8)
    print(f"Solved: {success}, Steps: {steps}")
    if success:
        print("Final board configuration:")
        for row in range(env.n):
            line = ""
            for col in range(env.n):
                if env.board[row] == col:
                    line += "Q "
                else:
                    line += ". "
            print(line)