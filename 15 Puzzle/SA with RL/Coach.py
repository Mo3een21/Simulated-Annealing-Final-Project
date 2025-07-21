import numpy as np
import time
from SquarePuzzle import SquarePuzzleEnv
from SAAgent import SAAgent

class Coach:
    def __init__(self, board_size=4, num_episodes=5000, max_steps=100000, learning_rate=0.0005, c=0.01):
        """
        Initializes the Coach to train an optimal weight matrix using gradient descent.
        """
        self.board_size = board_size
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.c = c  # From estimation

        self.weights = np.ones((board_size, board_size))

        # Tracking metrics
        self.success_count = 0
        self.loss_history = []
        self.steps_history = []

    def run_experiment(self, debug=False):
        """
        Runs one episode using SAAgent and returns success, steps taken, loss, and gradient.
        """
        env = SquarePuzzleEnv(self.board_size, weight_matrix=self.weights.copy())
        agent = SAAgent(env, max_steps=self.max_steps)

        trajectory = []

        # Run SAAgent while tracking last 1000 steps
        while agent.steps_taken < self.max_steps:
            if env.phi < 0.0001:  # Solution found
                break
            move = agent.step()
            trajectory.append((env.phi, move))
            
        success = (env.phi < 0.0001)
        steps_taken = agent.steps_taken

        if success:
            self.success_count += 1  # 🔥 Fix: Correctly count successes

        if not success or len(trajectory) < 10:
            return success, steps_taken, None, None  # Not enough data for learning

        # Take only the last 1000 steps
        last_trajectory = trajectory[-1000:]

        # Compute gradient and weights
        loss, grad = self.compute_loss_and_grad(last_trajectory)

        # Compute AVERAGE loss & gradient
        avg_loss = loss / len(last_trajectory)
        avg_grad = grad / len(last_trajectory)

        return success, steps_taken, avg_loss, avg_grad

    def compute_loss_and_grad(self, trajectory):
        """
        Computes the loss and gradient by undoing the last 1000 moves and updating the feature map dynamically.
        """
        loss = 0.0
        grad = np.zeros_like(self.weights)

        # Initialize board as the SOLVED state
        board = np.arange(1, self.board_size ** 2 + 1).reshape(self.board_size, self.board_size)
        board[-1, -1] = 0  # Correctly place the blank tile in the bottom-right corner

        # Find blank tile position
        blank_x, blank_y = self.board_size - 1, self.board_size - 1

        # Initialize feature map as **zeros everywhere**
        f = np.zeros_like(self.weights)

        # Process trajectory in REVERSE order (undoing moves)
        for t, (phi, move) in enumerate(reversed(trajectory)):  
            # **Check if move was rejected**
            if move is None:
                # Move was rejected, skip modifying board but still update loss and grad
                target = self.c * t
                error = phi - target
                loss += error ** 2
                grad += (2 * error * f)  # Keep f unchanged
                continue  # Skip move processing

            # **UNDO the move: Move the blank BACKWARD!**
            move_x, move_y = blank_x - move[0], blank_y - move[1]
            board[blank_x, blank_y], board[move_x, move_y] = board[move_x, move_y], board[blank_x, blank_y]

            # Get the tile being moved BACK
            moved_tile = board[blank_x, blank_y]  # The tile that was in the blank square

            # Compute where this tile belongs in the SOLVED state
            goal_x, goal_y = divmod(moved_tile - 1, self.board_size)

            # Update the blank tile position
            blank_x, blank_y = move_x, move_y

            # Update feature map at the GOAL position!
            f[goal_x, goal_y] = abs(goal_x - move_x) + abs(goal_y - move_y)

            # Compute loss and gradient
            target = self.c * t
            error = phi - target
            loss += error ** 2
            grad[goal_x, goal_y] += (2 * error * f[goal_x, goal_y])  # Only update this tile

        return loss, grad



    def apply_gradient_descent(self, avg_grad):
        """Applies gradient descent update to the weight matrix."""
        self.weights -= self.learning_rate * avg_grad

    def train(self):
        """
        Trains the weight matrix using gradient descent over multiple episodes.
        """
        for ep in range(self.num_episodes):
            success, steps, loss, avg_grad = self.run_experiment(debug=False)

            if success and avg_grad is not None:
                self.loss_history.append(loss)
                self.steps_history.append(steps)
                self.apply_gradient_descent(avg_grad)

            # Print progress **every 10 episodes**
            if ep % 10 == 0:
                success_rate = self.success_count / (ep + 1)
                avg_steps = np.mean(self.steps_history) if self.steps_history else None
                avg_loss = np.mean(self.loss_history) if self.loss_history else None
                print(f"Episode {ep+1}: Success Rate: {success_rate:.4f}, Avg Steps: {avg_steps}, Avg Loss: {avg_loss}")
                print("Current Weights:")
                print(self.weights)

        print("Final Weight Matrix:")
        print(self.weights)


# Run training
if __name__ == "__main__":
    coach = Coach(board_size=4, num_episodes=10000, max_steps=100000, learning_rate=0.0007, c=0.008)
    coach.train()
