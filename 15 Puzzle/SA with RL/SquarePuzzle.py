import numpy as np
import copy

class SquarePuzzleEnv:
    """
    Environment for the (n^2 - 1) sliding tile puzzle.
    Optionally uses a weight matrix W such that W[i][j] is the weight
    for the tile whose goal (target) position is (i, j).
    """

    def __init__(self, n, initial_state=None, weight_matrix=None):
        """
        Initialize an n x n sliding tile puzzle.
        
        Parameters:
        - n: Board size (e.g., n=3 for the 8-puzzle).
        - initial_state: Optional custom starting state; otherwise, a random solvable state is generated.
        - weight_matrix: Optional weight matrix W of shape (n, n). If None, defaults to all ones.
        """
        self.n = n
        self.goal_state = self._generate_goal_state()

        if weight_matrix is None:
            self.W = np.ones((n, n))
        else:
            self.W = weight_matrix

        if initial_state is None:
            self.state = self._generate_solvable_state()
        else:
            self.state = np.array(initial_state)

        self.blank_pos = tuple(np.argwhere(self.state == 0)[0])
        self.phi = self._compute_phi()  # Weighted Manhattan distance heuristic

    def _generate_goal_state(self):
        """Generates the goal state."""
        goal = np.arange(1, self.n**2).tolist() + [0]
        return np.array(goal).reshape(self.n, self.n)

    def _generate_solvable_state(self):
        """Generates a random solvable board."""
        while True:
            tiles = np.arange(self.n**2)
            np.random.shuffle(tiles)
            board = tiles.reshape(self.n, self.n)
            if self._is_solvable(board):
                return board

    def _is_solvable(self, board):
        """Checks if the given board is solvable based on inversion count and blank position."""
        flat_board = board.flatten()
        inversions = sum(
            1 for i in range(len(flat_board)) for j in range(i + 1, len(flat_board))
            if flat_board[i] and flat_board[j] and flat_board[i] > flat_board[j]
        )
        blank_row = np.argwhere(board == 0)[0][0] + 1
        if self.n % 2 == 1:
            return inversions % 2 == 0
        else:
            return (inversions + blank_row) % 2 == 0

    def _compute_phi(self):
        """
        Computes the weighted Manhattan distance heuristic.
        For each tile, its contribution is:
            weight * (|row - goal_row| + |col - goal_col|)
        """
        phi = 0
        for row in range(self.n):
            for col in range(self.n):
                tile = self.state[row, col]
                if tile == 0:
                    continue  # Ignore blank tile
                # Calculate the goal position for this tile
                goal_row, goal_col = divmod(tile - 1, self.n)
                # Weight for this tile comes from its goal position in W
                weight = self.W[goal_row, goal_col]
                phi += weight * (abs(row - goal_row) + abs(col - goal_col))
        return phi

    def display(self):
        """Prints the current board state and phi."""
        print("\n".join([" ".join([f"{num:2d}" if num != 0 else "  " for num in row])
                         for row in self.state]))
        print(f"φ (Weighted Manhattan Distance): {self.phi}\n")

    def get_valid_moves(self):
        """Returns a list of possible moves [(dx, dy)]."""
        x, y = self.blank_pos
        moves = []
        if x > 0: moves.append((-1, 0))  # UP
        if x < self.n - 1: moves.append((1, 0))  # DOWN
        if y > 0: moves.append((0, -1))  # LEFT
        if y < self.n - 1: moves.append((0, 1))  # RIGHT
        return moves

    def make_move(self, dx, dy):
        """
        Moves the blank tile and updates phi based on the weight matrix.

        Parameters:
        - dx, dy: Direction of the move.

        Returns:
        - move_info: The move (dx, dy), for compatibility.
        - delta_phi: Change in weighted heuristic.
        """
        x, y = self.blank_pos
        new_x, new_y = x + dx, y + dy
        tile = self.state[new_x, new_y]  # Tile that will move

        # Compute current contribution of the moving tile:
        goal_x, goal_y = divmod(tile - 1, self.n)
        weight = self.W[goal_x, goal_y]
        # Distance when tile is in the new position (since it moves to the blank spot)
        old_dist = abs(new_x - goal_x) + abs(new_y - goal_y)
        new_dist = abs(x - goal_x) + abs(y - goal_y)
        delta_phi = weight * (new_dist - old_dist)

        # Perform the move (swap tile and blank)
        self.state[x, y], self.state[new_x, new_y] = self.state[new_x, new_y], self.state[x, y]
        self.blank_pos = (new_x, new_y)
        self.phi += delta_phi

        return (dx, dy), delta_phi

    def unmake_move(self, move):
        """Undoes a move by applying its inverse."""
        dx, dy = move
        self.make_move(-dx, -dy)

    def clone(self):
        """Returns a deep copy of the environment."""
        return copy.deepcopy(self)
