import math
import random
from Game_Sudoku import SudokuEnv, generate_sudoku_puzzle
import sys
sys.stdout.reconfigure(encoding='utf-8')

class QLearnAgent:
   

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, epsilon_decay=0.99999):
        self.Q = [0.0] * 9
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
# ===================================================================
# this function chooses a subgrid based on epsilon-greedy strategy
    def choose_subgrid(self):
        if random.random() < self.epsilon:
            idx = random.randint(0, 8)
        else:
            maxQ = max(self.Q)
            candidates = [i for i, q in enumerate(self.Q) if q == maxQ]
            idx = random.choice(candidates)

        subgrid_row = idx // 3
        subgrid_col = idx % 3
        return (subgrid_row, subgrid_col), idx
# ===================================================================
# this function updates the Q-value for a given subgrid index based on the received reward
    def update(self, idx, reward):
        max_next = max(self.Q)
        self.Q[idx] = self.Q[idx] + self.alpha * (reward + self.gamma * max_next - self.Q[idx])

        self.epsilon *= self.epsilon_decay
        if self.epsilon < 0.01:
            self.epsilon = 0.01

# ====================================================================
# this class integrates Simulated Annealing with Reinforcement Learning to solve Sudoku puzzles
# ====================================================================
class SARLAgent:

    def __init__(self, env: SudokuEnv,
                 T0=0.5, cooling_rate=0.99999,
                 max_steps=2000000,
                 rl_alpha=0.1, rl_gamma=0.9, rl_epsilon=0.2):
        self.env = env
        self.T = T0
        self.cooling_rate = cooling_rate
        self.max_steps = max_steps
        self.steps_taken = 0
        self.rl_agent = QLearnAgent(alpha=rl_alpha, gamma=rl_gamma, epsilon=rl_epsilon)
        self.trajectory = []
# ===================================================================

    def step(self, debug=False):
        if self.env.phi == 0 or self.steps_taken >= self.max_steps:
            return
        (subgrid_row, subgrid_col), subgrid_idx = self.rl_agent.choose_subgrid()
        move_info, delta = self.env.make_move(chosen_subgrid=(subgrid_row, subgrid_col))

        if move_info is None:
            self.steps_taken += 1
            return

        reward = -delta
        self.rl_agent.update(subgrid_idx, reward)

        accept = False
        if delta <= 0:
            accept = True
        else:
            accept_prob = math.exp(-delta / self.T) if self.T > 0 else 0.0
            if random.random() < accept_prob:
                accept = True

        if debug:
            print(f"Step {self.steps_taken}, T={self.T:.6f}, φ={self.env.phi:.1f}")
            print(f"Chose subgrid ({subgrid_row},{subgrid_col}), Δφ={delta:.1f}, reward={reward:.1f}, ε={self.rl_agent.epsilon:.4f}")
            print(self.env.display())
            print("------")

        if accept:
            self.trajectory.append((move_info[0], move_info[1], self.env.phi))
            self.T *= self.cooling_rate ** 10
        else:
            self.env.unmake_move(move_info)
            self.T /= self.cooling_rate

        self.steps_taken += 1
# ===================================================================
    def run(self, debug=False):
        
        while self.steps_taken < self.max_steps:
            if self.env.phi == 0:
                return True
            self.step(debug=debug)
        return False

# ====================================================================
if __name__ == "__main__":
    initial_grid = generate_sudoku_puzzle(difficulty=30)

    env = SudokuEnv(initial_grid)
    print("Random Sudoku Puzzle:")
    print(env.display())
    print(f"\nInitial Conflicts (φ): {env.phi:.1f}")

    solver = SARLAgent(env,
                       T0=0.8,
                       cooling_rate=0.99999,
                       max_steps=10**6,
                       rl_alpha=0.1,
                       rl_gamma=0.9,
                       rl_epsilon=0.1)

    solved = solver.run(debug=False)

    if solved:
        print("\nSolved Sudoku:")
        print(env.display())
        print(f"\nSolution found in {solver.steps_taken} steps.")
    else:
        print("\nFailed to solve within step limit.")
        print("Final state:")
        print(env.display())
        print(f"\nRemaining Conflicts (φ): {env.phi:.1f}")
