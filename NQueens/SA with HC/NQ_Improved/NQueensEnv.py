import numpy as np

class NQueensEnv:
    def __init__(self, n, weights_col=None, weights_diag_plus=None, weights_diag_minus=None):
        self.n = n
        # The board is represented as an array of length n.
        # board[r] = c means that the queen in row r is in column c.
        self.board = np.zeros(n, dtype=int)
        
        # Conflict counters for columns and diagonals.
        # There are n columns, and 2*n-1 diagonals for each direction.
        self.conflicts_col = np.zeros(n, dtype=int)
        self.conflicts_diag_plus = np.zeros(2*n - 1, dtype=int)
        self.conflicts_diag_minus = np.zeros(2*n - 1, dtype=int)
        
        # Weight arrays for the linear potential function.
        # They have the same lengths as the corresponding conflict vectors.
        if weights_col is None:
            self.weights_col = np.ones(n, dtype=float)
            self.weights_diag_plus = np.ones(2*n - 1, dtype=float)
            self.weights_diag_minus = np.ones(2*n - 1, dtype=float)
        else:
            self.weights_col = weights_col
            self.weights_diag_plus = weights_diag_plus
            self.weights_diag_minus = weights_diag_minus
        
        # The current potential function value.
        self.phi = 0.0
        
        # Initialize a random board.
        self.init_board()
    
    # Helper functions to compute diagonal indices.
    def diag_plus_index(self, row, col):
        return row + col  # Range: 0 to 2*n-2
    
    def diag_minus_index(self, row, col):
        # Shift by (n-1) so that indices are nonnegative.
        return row - col + (self.n - 1)  # Range: 0 to 2*n-2

    def init_board(self):
        """Initialize the board by placing a random column for each row,
        and update conflict counters accordingly."""
        self.board = np.random.randint(0, self.n, size=self.n)
        self.conflicts_col.fill(0)
        self.conflicts_diag_plus.fill(0)
        self.conflicts_diag_minus.fill(0)
        
        for r in range(self.n):
            c = self.board[r]
            self.conflicts_col[c] += 1
            self.conflicts_diag_plus[self.diag_plus_index(r, c)] += 1
            self.conflicts_diag_minus[self.diag_minus_index(r, c)] += 1
            
        self.phi = self.compute_phi()
    
    def compute_phi(self):
        """Compute the overall potential function value from scratch.
        For each line, the feature is max(0, count-1), multiplied by its weight."""
        phi = 0.0
        # Columns
        for j in range(self.n):
            f_val = max(0, self.conflicts_col[j] - 1)
            phi += self.weights_col[j] * f_val
        # Positive-slope diagonals
        for idx in range(2*self.n - 1):
            f_val = max(0, self.conflicts_diag_plus[idx] - 1)
            phi += self.weights_diag_plus[idx] * f_val
        # Negative-slope diagonals
        for idx in range(2*self.n - 1):
            f_val = max(0, self.conflicts_diag_minus[idx] - 1)
            phi += self.weights_diag_minus[idx] * f_val
        return phi

    def update_line(self, conflicts_array, weights_array, index, delta):
        """
        Update a single conflict line by delta (+1 or -1).  
        Computes the change in the line’s contribution to phi.
        """
        old_count = conflicts_array[index]
        old_feature = max(0, old_count - 1)
        new_count = old_count + delta
        new_feature = max(0, new_count - 1)
        conflicts_array[index] = new_count
        # The change in potential for this line is weight * (new_feature - old_feature)
        return weights_array[index] * (new_feature - old_feature)
    
    def make_move(self, row, new_col):
        """
        Move the queen in a given row to a new column.
        This method updates the conflict counters and the potential phi in O(1).
        It returns a tuple (move_info, delta_phi) that can be used by unmake_move.
        """
        old_col = self.board[row]
        if old_col == new_col:
            return None  # No change
        
        delta_phi = 0.0
        # Remove the queen from its old position:
        # Update column
        delta_phi += self.update_line(self.conflicts_col, self.weights_col, old_col, -1)
        # Update positive diagonal
        dp_idx_old = self.diag_plus_index(row, old_col)
        delta_phi += self.update_line(self.conflicts_diag_plus, self.weights_diag_plus, dp_idx_old, -1)
        # Update negative diagonal
        dm_idx_old = self.diag_minus_index(row, old_col)
        delta_phi += self.update_line(self.conflicts_diag_minus, self.weights_diag_minus, dm_idx_old, -1)
        
        # Change the board: move the queen.
        self.board[row] = new_col
        
        # Add the queen at its new position:
        # Update column
        delta_phi += self.update_line(self.conflicts_col, self.weights_col, new_col, +1)
        # Update positive diagonal
        dp_idx_new = self.diag_plus_index(row, new_col)
        delta_phi += self.update_line(self.conflicts_diag_plus, self.weights_diag_plus, dp_idx_new, +1)
        # Update negative diagonal
        dm_idx_new = self.diag_minus_index(row, new_col)
        delta_phi += self.update_line(self.conflicts_diag_minus, self.weights_diag_minus, dm_idx_new, +1)
        
        # Update the overall potential
        self.phi += delta_phi
        
        # Pack the minimal info needed to reverse this move.
        move_info = (row, old_col, new_col, dp_idx_old, dp_idx_new, dm_idx_old, dm_idx_new)
        return move_info, delta_phi

    def unmake_move(self, move_info):
        """
        Undo a move given by move_info.
        This reverses the updates to the conflict counts and potential function.
        """
        row, old_col, new_col, dp_idx_old, dp_idx_new, dm_idx_old, dm_idx_new = move_info
        delta_phi = 0.0
        # Remove the queen from the new position:
        delta_phi += self.update_line(self.conflicts_col, self.weights_col, new_col, -1)
        delta_phi += self.update_line(self.conflicts_diag_plus, self.weights_diag_plus, dp_idx_new, -1)
        delta_phi += self.update_line(self.conflicts_diag_minus, self.weights_diag_minus, dm_idx_new, -1)
        # Restore the queen to its old position:
        self.board[row] = old_col
        delta_phi += self.update_line(self.conflicts_col, self.weights_col, old_col, +1)
        delta_phi += self.update_line(self.conflicts_diag_plus, self.weights_diag_plus, dp_idx_old, +1)
        delta_phi += self.update_line(self.conflicts_diag_minus, self.weights_diag_minus, dm_idx_old, +1)
        self.phi += delta_phi
        return delta_phi

    def get_feature_vector(self):
        """
        Return the combined feature vector (for columns, diag_plus, diag_minus)
        where each feature is max(0, conflict_count-1).
        """
        f_cols = np.maximum(0, self.conflicts_col - 1)
        f_diag_plus = np.maximum(0, self.conflicts_diag_plus - 1)
        f_diag_minus = np.maximum(0, self.conflicts_diag_minus - 1)
        return np.concatenate([f_cols, f_diag_plus, f_diag_minus])
    
    def gradient_for_state(self, target):
        """
        Compute the gradient for a single state.
        Let L = (phi - t)^2, so dL/dw = 2*(phi - t)*f,
        where f is the feature vector.
        """
        features = self.get_feature_vector()  # shape: (n + 2*(2*n-1),)
        error = self.phi - target
        grad = 2 * error * features
        return grad

    def update_weights(self, grad, learning_rate):
        """
        Update the weight arrays using the gradient vector.
        The gradient vector is assumed to be ordered as:
          - first n entries for columns,
          - next 2*n-1 for diag_plus,
          - last 2*n-1 for diag_minus.
        """
        n_cols = self.n
        n_diags = 2*self.n - 1
        self.weights_col -= learning_rate * grad[:n_cols]
        self.weights_diag_plus -= learning_rate * grad[n_cols:n_cols+n_diags]
        self.weights_diag_minus -= learning_rate * grad[n_cols+n_diags:]

    def __str__(self):
        # Returns a string representation of the board as an n x n matrix.
        s = ""
        for r in range(self.n):
            row_str = ""
            for c in range(self.n):
                if self.board[r] == c:
                    row_str += "Q "
                else:
                    row_str += ". "
            s += row_str.rstrip() + "\n"
        return s

    def display(self):
        # Simply returns the same as __str__.
        return str(self)
    
def main():
    np.random.seed(42)  # For reproducibility
    # Initialize environment
    n = 5

    # Assign random weights (positive values)
    env = NQueensEnv(n, np.random.uniform(0.1, 2.0, size=n), np.random.uniform(0.1, 2.0, size=2 * n - 1), np.random.uniform(0.1, 2.0, size=2 * n - 1))


    print("\nInitial Weights:")
    print("Column Weights:", env.weights_col)
    print("Diagonal Plus Weights:", env.weights_diag_plus)
    print("Diagonal Minus Weights:", env.weights_diag_minus)

    print("\nStarting Debugging Moves...")
    
    for i in range(50):  # Perform 50 random moves
        row = np.random.randint(0, n)
        new_col = np.random.randint(0, n)

        if env.board[row] == new_col:
            continue  # Skip if the move is the same

        print("Board State:")
        print(env)

        print("Phi = ", env.phi)
        print("Conflict Vector: ")
        print(env.conflicts_col)
        print(env.conflicts_diag_plus)
        print(env.conflicts_diag_minus)
        print(env.get_feature_vector())

        move_info, delta_phi = env.make_move(row, new_col)

if __name__ == "__main__":
    main()

