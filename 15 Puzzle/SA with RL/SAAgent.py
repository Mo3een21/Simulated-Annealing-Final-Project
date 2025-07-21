import math
import numpy as np
import random
import time
from SquarePuzzle import SquarePuzzleEnv

class SAAgent:
    """
    Simulated Annealing (SA) Agent for solving the (n^2 - 1) sliding tile puzzle.
    """

    def __init__(self, env, T0=0.5, cooling_rate=0.95, heating_rate=1.05, max_steps=100000):
        """
        Initialize the SA agent.

        Parameters:
        - env: An instance of SquarePuzzleEnv.
        - T0: Initial temperature.
        - cooling_rate: Multiplicative cooling factor (e.g., 0.99).
        - heating_rate: Multiplicative heating factor (e.g., 1.01).
        - max_steps: Maximum number of moves allowed.
        """
        self.env = env
        self.T = T0
        self.cooling_rate = cooling_rate
        self.heating_rate = heating_rate
        self.max_steps = max_steps
        self.steps_taken = 0

    def step(self, debug=False):
        """
        Perform one step of simulated annealing.
        """
        valid_moves = self.env.get_valid_moves()

        # Choose a random valid move (dx, dy)
        move = random.choice(valid_moves)

        # Get current temperature
        T = self.T

        if debug:
            print(f"Step {self.steps_taken}, Temperature: {T:.4f}")
            print(f"Attempting move: {move}")
            print("Board before move:")
            self.env.display()
            print(f"Phi before move: {self.env.phi:.4f}")

        # Make the move and compute energy difference
        _, delta_phi = self.env.make_move(*move)

        # SA Acceptance Rule
        if delta_phi <= 0:
            accept = True
        else:
            accept_prob = math.exp(-delta_phi / T) if T > 0 else 0.0
            accept = random.random() < accept_prob

        if accept:
            self.T *= self.cooling_rate  # Cooling step
            if debug:
                print(f"Move accepted (Δφ = {delta_phi:.4f}). New phi: {self.env.phi:.4f}")
        else:
            self.env.unmake_move(move)  # Reject move
            self.T *= self.heating_rate  # Reheating step
            if debug:
                print(f"Move rejected (Δφ = {delta_phi:.4f}). Phi remains: {self.env.phi:.4f}")

        if debug:
            print("Board after move:")
            self.env.display()
            print("-----")

        self.steps_taken += 1
        return move if accept else None

    def run(self, debug=False):
        """
        Run the SA agent until a solution is found (phi = 0) or max_steps is reached.
        """
        while self.steps_taken < self.max_steps:
            if self.env.phi < 0.0001:
                return True  # Solution found
            self.step(debug=debug)
        return False  # Step limit reached

weights = np.ones((4,4))
weights[0:3][0] = 2
weights[0][0:3] = 2

print('Hi')

avg = 0
for i in range(100):
    env = SquarePuzzleEnv(n=4, weight_matrix=weights)
    agent = SAAgent(env)
    agent.run()
    avg += agent.steps_taken
avg = avg / 100
print("Steps taken: ", avg)
