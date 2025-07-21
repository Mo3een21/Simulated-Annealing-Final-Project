import math
import random
from NQueensEnv import NQueensEnv

class SAAgent:
    def __init__(self, env, T0=1.0, cooling_rate=0.999, max_steps=10000):
        """
        Initialize the agent using classical simulated annealing.
        
        Parameters:
          env         : an instance of NQueensEnv.
          T0          : initial temperature.
          cooling_rate: multiplicative cooling factor (alpha). For example, 0.999.
          max_steps   : maximum number of moves in one episode.
        """
        self.env = env
        self.T = T0
        self.cooling_rate = cooling_rate
        self.max_steps = max_steps
        self.steps_taken = 0
        self.trajectory = []  # Each accepted move is recorded as (row, old_col, new_col, phi_after_move).

    def step(self, debug=False):
        """Perform one step of classical simulated annealing.
        
           When debug is True, print detailed information about the move.
        """
        n = self.env.n
        # Pick a random row.
        row = random.randint(0, n - 1)
        current_col = self.env.board[row]
        # Pick a new column (different from current).
        new_col = current_col
        while new_col == current_col:
            new_col = random.randint(0, n - 1)
        
        # Get current temperature.
        T = self.T
        
        if debug:
            print(f"Step {self.steps_taken}, Temperature: {T:.4f}")
            print(f"Attempting move in row {row} from column {current_col} to {new_col}")
            print("Board before move:")
            print(self.env.display())
            print(f"Phi before move: {self.env.phi:.4f}")
        
        # Try to make the move.
        move_result = self.env.make_move(row, new_col)
        if move_result is None:
            if debug:
                print("No move executed (new column equals current column).")
            return
        move_info, delta_phi = move_result
        
        # Classical SA acceptance:
        if delta_phi <= 0:
            accept = True
        else:
            accept_prob = math.exp(-delta_phi / T) if T > 0 else 0.0
            accept = random.random() < accept_prob
        
        if accept:
            self.trajectory.append((row, current_col, new_col, self.env.phi))
            self.T *= (self.cooling_rate) ** 10
            if debug:
                print(f"Move accepted (Δφ = {delta_phi:.4f}). New phi: {self.env.phi:.4f}")
        else:
            self.T /= self.cooling_rate
            self.env.unmake_move(move_info)
            if debug:
                print(f"Move rejected (Δφ = {delta_phi:.4f}). Phi remains: {self.env.phi:.4f}")
        
        if debug:
            print("Board after move:")
            print(self.env.display())
            print("-----")
        
        self.steps_taken += 1

    def run(self, debug=False):
        """Run the SA agent until a solution is found (phi = 0) or max_steps is reached.
        
           If debug is True, each step is printed in detail.
        """
        while self.steps_taken < self.max_steps:
            if self.env.phi < 0.01:
                return True
            self.step(debug=debug)
        return False
