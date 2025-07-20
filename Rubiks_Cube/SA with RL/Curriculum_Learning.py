import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
from Rubiks_Cube import RubiksCubeEnv  
from SAAgent import SAAgent
from DQN import DQN
# ==============================================================================
# Experience Replay Buffer
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# Reinforcement Learning Agent with Curriculum Learning
# ==============================================================================
class RLAgent:
    def __init__(self, state_size, action_size, lr=1e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(capacity=50000)
        
        # Neural networks
        self.q_network = DQN(state_size, output_size=action_size)
        self.target_network = DQN(state_size, output_size=action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_freq = 100
        self.steps = 0
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.stage_success_threshold = 0.8
        self.stage_attempts = 0
        self.stage_successes = 0
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.detach().numpy())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select action, target network to evaluate
        next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_curriculum(self, success):
        """Update curriculum learning stage based on performance"""
        self.stage_attempts += 1
        if success:
            self.stage_successes += 1
        
        # Check if we should advance to next stage
        if self.stage_attempts >= 50:
            success_rate = self.stage_successes / self.stage_attempts
            if success_rate >= self.stage_success_threshold:
                self.curriculum_stage += 1
                print(f"Advancing to curriculum stage {self.curriculum_stage}")
            
            # Reset counters
            self.stage_attempts = 0
            self.stage_successes = 0
    
    def get_shuffle_steps(self):
        """Get number of shuffle steps based on curriculum stage"""
        # Start with easy problems and gradually increase difficulty
        stages = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
        if self.curriculum_stage < len(stages):
            return stages[self.curriculum_stage]
        return stages[-1]
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'curriculum_stage': self.curriculum_stage
        }, filepath)
    
    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.curriculum_stage = checkpoint['curriculum_stage']

# ==============================================================================
# Enhanced Simulated Annealing with Learned Q-Values
# ==============================================================================
class RLSimulatedAnnealing:
    def __init__(self, env, agent, T0=1.0, cooling_rate=0.999, use_learned_policy=True):
        self.env = env
        self.agent = agent
        self.T = T0
        self.cooling_rate = cooling_rate
        self.use_learned_policy = use_learned_policy
        self.trajectory = []
        
    def get_move_probabilities(self, state):
        """Get move probabilities based on Q-values and temperature"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.agent.q_network(state_tensor).squeeze().detach().numpy()
        
        if self.use_learned_policy:
            # Use softmax with temperature to convert Q-values to probabilities
            scaled_q = q_values / max(self.T, 0.1)
            exp_q = np.exp(scaled_q - np.max(scaled_q))  # Subtract max for numerical stability
            probabilities = exp_q / np.sum(exp_q)
        else:
            # Uniform probabilities (random policy)
            probabilities = np.ones(len(q_values)) / len(q_values)
        
        return probabilities
    
    def step(self, training=True):
        """Execute one step of the algorithm"""
        state = self.env.get_state_vector()
        old_phi = self.env.phi
        
        if self.use_learned_policy and not training:
            # During evaluation, use greedy policy
            action = self.agent.act(state, training=False)
        else:
            # During training or with random policy, use probabilistic selection
            probabilities = self.get_move_probabilities(state)
            action = np.random.choice(len(probabilities), p=probabilities)
        
        move = self.env.moves[action]
        move_info, delta_phi = self.env.make_move(move)
        
        # Calculate reward
        reward = self.env.get_reward(old_phi)
        done = self.env.is_solved()
        
        # Store experience for training
        if training:
            next_state = self.env.get_state_vector()
            self.agent.remember(state, action, reward, next_state, done)
        
        # Simulated annealing acceptance criteria
        if delta_phi <= 0:
            accept = True
        else:
            accept_prob = math.exp(-delta_phi / self.T) if self.T > 0 else 0.0
            accept = random.random() < accept_prob
        
        if accept:
            self.trajectory.append((move, self.env.phi))
            self.T *= self.cooling_rate
        else:
            self.env.unmake_move(move_info)
            self.T /= self.cooling_rate
        
        return done
    
    def solve(self, max_steps=1000, training=True):
        """Try to solve the cube"""
        self.trajectory = []
        steps = 0
        
        while steps < max_steps and not self.env.is_solved():
            done = self.step(training)
            steps += 1
            
            if done:
                return True, steps
        
        return False, steps

# ==============================================================================
# Training Loop with Curriculum Learning
# ==============================================================================
def train_agent(episodes=1000, save_interval=100):
    """Train the RL agent using curriculum learning"""
    # Calculate state size
    dummy_env = RubiksCubeEnv(shuffle_steps=0)
    state_size = len(dummy_env.get_state_vector())
    action_size = len(dummy_env.moves)
    
    # Initialize agent
    agent = RLAgent(state_size, action_size, lr=1e-4)
    
    # Training statistics
    success_history = []
    steps_history = []
    
    print("Starting training with curriculum learning...")
    
    for episode in range(episodes):
        # Create environment with appropriate difficulty
        shuffle_steps = agent.get_shuffle_steps()
        env = RubiksCubeEnv(shuffle_steps=shuffle_steps)
        
        # Create SA solver with learned policy
        solver = RLSimulatedAnnealing(env, agent, T0=0.5, cooling_rate=0.995)
        
        # Attempt to solve
        solved, steps = solver.solve(max_steps=200, training=True)
        
        # Update curriculum
        agent.update_curriculum(solved)
        
        # Train the network
        if len(agent.memory) > agent.batch_size:
            for _ in range(10):  # Multiple training steps per episode
                agent.replay()
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record statistics
        success_history.append(1 if solved else 0)
        steps_history.append(steps)
        
        # Print progress
        if episode % 10 == 0:
            recent_success = np.mean(success_history[-50:]) if len(success_history) >= 50 else np.mean(success_history)
            print(f"Episode {episode}, Shuffle Steps: {shuffle_steps}, "
                  f"Success Rate: {recent_success:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Stage: {agent.curriculum_stage}")
        
        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f'rubiks_model_episode_{episode}.pth')
    
    return agent, success_history, steps_history

# ==============================================================================
# Evaluation Function
# ==============================================================================
def evaluate_agent(agent, num_tests=50, shuffle_steps=20):
    """Evaluate trained agent performance"""
    print(f"\nEvaluating agent on {num_tests} cubes with {shuffle_steps} shuffle steps...")
    
    successes = 0
    total_steps = 0
    
    for i in range(num_tests):
        env = RubiksCubeEnv(shuffle_steps=shuffle_steps)
        solver = RLSimulatedAnnealing(env, agent, T0=0.1, cooling_rate=0.99, use_learned_policy=True)
        
        solved, steps = solver.solve(max_steps=500, training=False)
        
        if solved:
            successes += 1
            total_steps += steps
            print(f"Test {i+1}: Solved in {steps} steps")
        else:
            print(f"Test {i+1}: Failed to solve")
    
    success_rate = successes / num_tests
    avg_steps = total_steps / successes if successes > 0 else 0
    
    print(f"\nResults: {successes}/{num_tests} solved ({success_rate:.1%})")
    if successes > 0:
        print(f"Average steps to solution: {avg_steps:.1f}")
    
    return success_rate, avg_steps

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # Train the agent
    print("Training Rubik's Cube solver with Reinforcement Learning...")
    agent, success_history, steps_history = train_agent(episodes=500)
    
    # Save final model
    agent.save_model('rubiks_final_model.pth')
    
    # Evaluate on different difficulty levels
    for shuffle_steps in [5, 10, 15, 20]:
        evaluate_agent(agent, num_tests=20, shuffle_steps=shuffle_steps)
    
    # Compare with classical SA
    print("\n\nComparing with Classical Simulated Annealing:")
    classical_successes = 0
    
    for i in range(20):
        env = RubiksCubeEnv(shuffle_steps=10)
        classical_agent = SAAgent(env, T0=0.8, cooling_rate=0.999, max_steps=10000)
        solved = classical_agent.run()
        if solved:
            classical_successes += 1
    
    print(f"Classical SA: {classical_successes}/20 solved ({classical_successes/20:.1%})")