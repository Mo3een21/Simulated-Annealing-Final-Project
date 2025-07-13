import math
import random
from Game_Sudoku import SudokuEnv, generate_sudoku_puzzle
class SAAgent:
    def __init__(self, env, T0=0.5, cooling_rate=0.999, max_steps=10000):
        
        self.env = env
        self.T = T0
        self.cooling_rate = cooling_rate
        self.max_steps = max_steps
        self.steps_taken = 0
        self.trajectory = [] 

    def step(self, debug=False):
        if self.env.phi == 0 or self.steps_taken >= self.max_steps:
            return

        if debug:
            print(f"Step {self.steps_taken}, Temperature: {self.T:.4f}")
            print("Board before move:")
            print(self.env.display())
            print(f"Current conflicts: {self.env.phi}")

        # Generate potential move
        move_info, delta = self.env.make_move()
        if move_info is None:  # No valid move possible
            if debug:
                print("No valid moves in selected subgrid")
            return

        pos1, pos2 = move_info
        (r1, c1), (r2, c2) = pos1, pos2

        # Classical SA acceptance
        if delta <= 0:
            accept = True
        else:
            accept_prob = math.exp(-delta / self.T) if self.T > 0 else 0.0
            accept = random.random() < accept_prob

        if debug:
            print(f"Attempting swap: ({r1},{c1}) <-> ({r2},{c2})")
            print(f"Δφ = {delta}, Accept probability: {math.exp(-delta/self.T) if delta>0 else 1.0:.4f}")

        if accept:
            self.trajectory.append((pos1, pos2, self.env.phi))
            self.T *= self.cooling_rate ** 10  # Faster cooling on acceptance
            if debug:
                print(f"Move accepted. New conflicts: {self.env.phi}")
        else:
            self.T /= self.cooling_rate  # Reverse cooling on rejection
            self.env.unmake_move(move_info)
            if debug:
                print(f"Move rejected. Conflicts remain: {self.env.phi}")

        if debug:
            print("Board after move:")
            print(self.env.display())
            print("-----")

        self.steps_taken += 1

    def run(self, debug=False):
        """Run the SA agent until solution is found or max_steps reached."""
        while self.steps_taken < self.max_steps:
            if self.env.phi < 0.01:
                return True
            self.step(debug=debug)
        return False

if __name__ == "__main__":
    # Generate new random puzzle with 45 empty cells
    initial_grid = generate_sudoku_puzzle(difficulty=50)
    
    env = SudokuEnv(initial_grid)
    print("Random Sudoku Puzzle:")
    print(env.display())
    print(f"\nInitial Conflicts: {env.phi}")

    solver = SAAgent(env, T0=0.3, cooling_rate=0.99999, max_steps=10**6)
    if solver.run():
        print("\nSolved Sudoku:")
        print(env.display())
        print(f"\nSolution found in {solver.steps_taken} steps")
    else:
        print("\nFailed to solve Sudoku")
        print("Final state:")
        print(env.display())
        