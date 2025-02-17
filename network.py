import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Neural Network definition
class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc4(x)

class Board:
    
    def __init__(self, n):
        self.n = n
        self.reset()
        
    def reset(self):
        self.queen_cols = [random.randint(0, n - 1) for _ in range(n)]
        self.cols, self.plus_diags, self.minus_diags, self.conflicts = self.count_conflicts()

    def count_conflicts(self):
        n = self.n
        cols = [0] * n
        plus_diags = [0] * (2 * n - 1)
        minus_diags = [0] * (2 * n - 1)
        conflicts = 0

        for row in range(n):
            col = self.queen_cols[row]
            conflicts += cols[col] + plus_diags[row + col] + minus_diags[row - col + n - 1]
            cols[col] += 1
            plus_diags[row + col] += 1
            minus_diags[row - col + n - 1] += 1

        return cols, plus_diags, minus_diags, conflicts
    
    def move_queen(self, row, new_col):
        n = self.n
        delta_conflicts = 0
        
        old_col = self.queen_cols[row]
        self.queen_cols[row] = new_col

        self.cols[old_col] -= 1
        self.plus_diags[row + old_col] -= 1
        self.minus_diags[row - old_col + n - 1] -= 1
        delta_conflicts -= self.cols[old_col] + self.plus_diags[row + old_col] + self.minus_diags[row - old_col + n - 1]
        
        delta_conflicts += self.cols[new_col] + self.plus_diags[row + new_col] + self.minus_diags[row - new_col + n - 1]
        self.cols[new_col] += 1
        self.plus_diags[row + new_col] += 1
        self.minus_diags[row - new_col + n - 1] += 1
        
        self.conflicts += delta_conflicts
        return delta_conflicts


    def encode_state(self):
        """Encode board state with proper dimensions"""
        n = self.n
        board_state = np.zeros(n * n)
        for row, col in enumerate(self.queen_cols):
            board_state[row * n + col] = 1
        return np.concatenate([
            board_state,
            self.cols,
            self.plus_diags,
            self.minus_diags
        ])

    def decode_action(self, action):
        """Decode an action index into a (row, col) tuple."""
        # action is place on tile, for example, action =17 means its on the 17th tile(3rd row, second column)
        n = self.n
        row = action // n
        col = action % n
        return row, col
    
    def print_board(self):
        """Print the board."""
        n = self.n
        for row in range(n):
            for col in range(n):
                if col == self.queen_cols[row]:
                    print("Q", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()
    

# Simulated Annealing with Q-Network
class QLearningAgent:
    
    def __init__(self, n, epsilon=0.04, decay_rate=0.99, network=None):
        self.n = n
        self.board = Board(n)
        if network is None:
            self.network = QNetwork(n_inputs=n*n*len(self.board.encode_state()), n_outputs=n * n)
        else:
            self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def get_q_values(self, state):
        """Return Q-values for a given state."""
        state_tensor = torch.tensor(state, dtype=torch.float32 , requires_grad=True).unsqueeze(0)
        q_values = self.network(state_tensor)
        return q_values.squeeze(0).detach().numpy()

    def choose_action(self, q_values):
        """Choose an action using epsilon-greedy exploration."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n * self.n - 1)
        return np.argmax(q_values)

    # =========================================================================================
    def update_network(self, q_values, action, reward, next_q_values):
       """Train the Q-network with a single step."""
       # Convert state and next_state to tensors
    #    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
   
       # Convert q_values and next_q_values to tensors
       q_values_tensor = torch.tensor(q_values, dtype=torch.float32, requires_grad=True)
       next_q_values_tensor = torch.tensor(next_q_values, dtype=torch.float32, requires_grad=True)
   
         # Update the Q-value of the chosen action
       target_q_values = q_values_tensor.clone().detach()
       target_q_values[action] = reward + 0.9 * torch.max(next_q_values_tensor)
   
       # Calculate loss
       loss = self.criterion(q_values_tensor, target_q_values)
   
       # Backpropagation and optimization step
       self.optimizer.zero_grad()
       loss.backward()
       self.optimizer.step()

    
    
    
    # ========================================
    def train(self, iterations=1000, max_steps=500):
        """Train the Q-network using epsilon-greedy policy."""
        
        success = 0
        for i in range(iterations):
            self.board.reset()
            state = self.board.encode_state()
            q_values = self.get_q_values(state)
            tmp_epsilon = self.epsilon
            for step in range(max_steps):
                occupied_positions = np.where(self.board.encode_state() == 1)[0]
                q_values_tmp = q_values.copy()
                q_values_tmp[occupied_positions] = -np.inf
                action = self.choose_action(q_values_tmp)
                prev_state = state.copy()
                row, col = self.board.decode_action(action)
                
                delta_conflicts = self.board.move_queen(row, col)
                state = self.board.encode_state()

                reward = -delta_conflicts - 1 if self.board.conflicts > 0 else 100 
            
                next_q_values = self.get_q_values(state)
                
                self.update_network(q_values, action, reward, next_q_values)
                q_values = next_q_values

                if self.board.conflicts == 0:
                    success += 1
                    break

                tmp_epsilon *= self.decay_rate
                
            print(f"Iteration {i + 1}: {step + 1} steps, conflicts: {self.board.conflicts}, successes: {success}")

    
n = 8
# Training the agent
model = torch.load("model.pth")

agent = QLearningAgent(n, network=model)

agent.train()
# print board
agent.board.print_board()


save_path = "model.pth"
torch.save(agent.network, save_path)
