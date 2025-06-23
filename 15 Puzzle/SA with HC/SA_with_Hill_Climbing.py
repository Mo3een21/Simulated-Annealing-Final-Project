import math
import random
# import to class Puzzle15Env
from Game_15Puzzle import Puzzle15Env

# ==============================================================================
# Simulated Annealing with hill climbing for the 15-puzzle problem
# ==============================================================================
class SAAgent:
    def __init__(self, env, T0=1.0, cooling_rate=0.999, max_steps=10000):
        self.env = env
        self.T = T0
        self.cooling_rate = cooling_rate
        self.max_steps = max_steps
        self.steps_taken = 0
        self.trajectory = [] 

    def step(self, debug=False):
        if self.env.phi == 0 or self.steps_taken >= self.max_steps:
            return

        valid_moves = self.env.get_valid_moves()
        if not valid_moves:
            return  # Should not happen in a valid puzzle state

        tile_pos = random.choice(valid_moves)

        if debug:
            print(f"Step {self.steps_taken}, Temperature: {self.T:.4f}")
            print(f"Attempting to move tile from position {tile_pos} (blank is at {self.env.blank_pos})")
            print("Board before move:")
            print(self.env.display())
            print(f"Phi before move: {self.env.phi}")

        old_phi = self.env.phi
        move_info, delta_phi = self.env.make_move(tile_pos)

        # Decide acceptance
        if delta_phi <= 0:
            accept = True
        else:
            accept_prob = math.exp(-delta_phi / self.T) if self.T > 0 else 0.0
            accept = random.random() < accept_prob

        if accept:
            self.trajectory.append((tile_pos, move_info[1], self.env.phi))
            self.T *= (self.cooling_rate)
            if debug:
                print(f"Move accepted (Δφ = {delta_phi:.4f}). New phi: {self.env.phi}")
        else:
            self.env.unmake_move(move_info)
            self.T /= self.cooling_rate
            if debug:
                print(f"Move rejected (Δφ = {delta_phi:.4f}). Phi remains: {old_phi}")

        if debug:
            print("Board after move:")
            print(self.env.display())
            print("-----")

        self.steps_taken += 1

    def run(self, debug=False):
        while self.steps_taken < self.max_steps:
            if self.env.phi < 0.01:  
                return True
            self.step(debug=debug)
        return False

if __name__ == "__main__":
    # Initialize environment
    env = Puzzle15Env(shuffle_steps=1000)
    print("Initial Puzzle State:")
    print(env.display())
    print(f"\nInitial Heuristic Value: {env.phi}\n")
    
    # Run solver
    agent = SAAgent(env, T0=0.8, cooling_rate=0.999, max_steps=100000)
    solved = agent.run()
    
    # Print results
    print("\nFinal Puzzle State:")
    print(env.display())
    print(f"\nSteps Taken: {agent.steps_taken}")
    print(f"Final Heuristic Value: {env.phi}")
    print("Status:", "Solved!" if solved else "Solution not found")