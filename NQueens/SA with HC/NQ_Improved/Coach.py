import numpy as np
import random
import matplotlib.pyplot as plt
from NQueensEnv import NQueensEnv
from SAAgent import SAAgent

class Coach:
    def __init__(self, board_size, num_episodes, max_steps=5000, learning_rate=0.001, c=0.02):
        """
        board_size: board dimension (and number of queens)
        num_episodes: total training episodes
        max_steps: maximum moves per episode (5000 as per tuner success)
        learning_rate: learning rate for gradient descent updates (applied to weights)
        c: scaling constant for the targets (phi should approximate c*(T-i))
        """
        self.board_size = board_size
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.c = c
        
        # Create an initial environment and extract initial weights
        self.env = NQueensEnv(board_size)
        
        # Store the weight arrays (will be copied into each new environment)
        self.weights = {
            'col': self.env.weights_col.copy(),
            'diag_plus': self.env.weights_diag_plus.copy(),
            'diag_minus': self.env.weights_diag_minus.copy()
        }
        
        # Metrics for tracking training progress
        self.success_count = 0
        self.loss_history = []    # Losses from successful episodes
        self.steps_history = []   # Steps for successful episodes

    def run_experiment(self, debug=False):
        """
        Runs one episode from scratch using a fresh environment and SAAgent.
        Returns:
            - success (bool): whether a solution was found.
            - steps_taken: number of moves performed.
            - avg_loss: average squared error loss (if successful, else None).
            - T: total number of accepted moves (trajectory length).
            - trajectory: the recorded moves.
            - initial_board: the board configuration at the start.
            - avg_grad: the computed average gradient vector (if successful, else None).
        """
        # Create a fresh environment **with the correct weights**
        env = NQueensEnv(
            self.board_size,
            weights_col=self.weights['col'].copy(),
            weights_diag_plus=self.weights['diag_plus'].copy(),
            weights_diag_minus=self.weights['diag_minus'].copy()
        )

        initial_board = env.board.copy()
        
        # Instantiate SAAgent with T0=0.8 and cooling_rate=0.998.
        agent = SAAgent(env, T0=0.8, cooling_rate=0.998, max_steps=self.max_steps)
        agent.trajectory = []
        
        while agent.steps_taken < agent.max_steps:
            if env.phi < 0.001:  # Tolerance for floating point errors
                break
            agent.step(debug=debug)
        
        success = (env.phi < 0.001)
        steps_taken = agent.steps_taken
        
        if not success:
            return success, steps_taken, None, None, agent.trajectory, initial_board, None
        
        self.success_count += 1
        T = len(agent.trajectory)
        total_loss = 0.0
        grad_accum = np.zeros(len(env.get_feature_vector()))
        # Replay the trajectory in a fresh environment
        temp_env = NQueensEnv(
            self.board_size,
            weights_col=self.weights['col'].copy(),
            weights_diag_plus=self.weights['diag_plus'].copy(),
            weights_diag_minus=self.weights['diag_minus'].copy()
        )

        # Set board state
        temp_env.board = initial_board.copy()

        # Reset conflicts
        temp_env.conflicts_col.fill(0)
        temp_env.conflicts_diag_plus.fill(0)
        temp_env.conflicts_diag_minus.fill(0)

        # Recompute conflicts from the initial board
        for r in range(self.board_size):
            c = temp_env.board[r]
            temp_env.conflicts_col[c] += 1
            temp_env.conflicts_diag_plus[r + c] += 1
            temp_env.conflicts_diag_minus[r - c + (self.board_size - 1)] += 1

        # Now compute phi correctly
        temp_env.phi = temp_env.compute_phi()

        
        # Modified target: instead of target = T - i, we use target = (T-i)/50.
        # Only look at the last 100 steps.
        for i, (row, old_col, new_col, phi_recorded) in enumerate(agent.trajectory):
            f = temp_env.get_feature_vector()
            target = self.c * (T - i)  # Scaling factor for target
            error = phi_recorded - target
            total_loss += error**2
            grad_accum += 2 * error * f
            temp_env.make_move(row, new_col)
        
        avg_loss = total_loss / T
        avg_grad = grad_accum / T
        
        return success, steps_taken, avg_loss, T, agent.trajectory, initial_board, avg_grad

    def apply_gradient_descent(self, avg_grad):
        """
        Uses the average gradient to update the stored weight arrays.
        """
        n = self.board_size
        n_diags = 2 * self.board_size - 1
        self.weights['col'] -= self.learning_rate * avg_grad[:n]
        self.weights['diag_plus'] -= self.learning_rate * avg_grad[n:n+n_diags]
        self.weights['diag_minus'] -= self.learning_rate * avg_grad[n+n_diags:]
    
    def train(self):
        """
        The main training loop.
        For each episode, run an experiment and, if successful, update the weights.
        Every 100 episodes, print the success rate, average steps, average loss, and current weight vectors.
        Also plot charts of success rate and average steps over time.
        """
        bin_size = 200
        episodes_bins = []
        success_rate_bins = []
        avg_steps_bins = []
        
        for ep in range(self.num_episodes):
            success, steps, loss, T, trajectory, init_board, avg_grad = self.run_experiment(debug=False)
            if success:
                self.loss_history.append(loss)
                self.steps_history.append(steps)
                self.apply_gradient_descent(avg_grad)
            
            # Every bin_size episodes, compute and print aggregated metrics.
            if (ep + 1) % bin_size == 0:
                current_success_rate = (self.success_count / (ep + 1)) * 100
                current_avg_steps = np.mean(self.steps_history) if self.steps_history else None
                current_avg_loss = np.mean(self.loss_history) if self.loss_history else None
                episodes_bins.append(ep + 1)
                success_rate_bins.append(current_success_rate)
                avg_steps_bins.append(current_avg_steps)
                print(f"After {ep+1} episodes:")
                print(f"  Success Rate: {current_success_rate:.2f}%")
                print(f"  Average Steps (successful): {current_avg_steps}")
                print(f"  Average Loss (successful): {current_avg_loss}")
                print("  Current weight vectors:")
                print("    col:", self.weights['col'])
                print("    diag_plus:", self.weights['diag_plus'])
                print("    diag_minus:", self.weights['diag_minus'])
        
        overall_success_rate = (self.success_count / self.num_episodes) * 100
        overall_avg_loss = np.mean(self.loss_history) if self.loss_history else None
        overall_avg_steps = np.mean(self.steps_history) if self.steps_history else None
        print("\nTraining complete.")
        print("Final Metrics:")
        print(f"  Overall Success Rate: {overall_success_rate:.2f}%")
        if overall_avg_loss is not None:
            print(f"  Average Loss (successful episodes): {overall_avg_loss}")
        if overall_avg_steps is not None:
            print(f"  Average Steps (successful episodes): {overall_avg_steps}")
        print("Final learned weight vectors:")
        print("  col:", self.weights['col'])
        print("  diag_plus:", self.weights['diag_plus'])
        print("  diag_minus:", self.weights['diag_minus'])
        
        # Plot charts.
        plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(episodes_bins, success_rate_bins, marker='o')
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate (%)")
        plt.title("Success Rate over Training")
        
        plt.subplot(1,2,2)
        plt.plot(episodes_bins, avg_steps_bins, marker='o', color='orange')
        plt.xlabel("Episodes")
        plt.ylabel("Average Steps (Successful Episodes)")
        plt.title("Average Steps over Training")
        plt.tight_layout()
        plt.show()

def main():
    coach = Coach(board_size=8, num_episodes=5000, max_steps=10000, learning_rate=0.001, c=0.02)
    coach.train()

if __name__ == "__main__":
    main()
