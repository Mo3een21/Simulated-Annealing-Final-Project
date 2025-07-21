import random

import numpy as np

def initialize_board(n):
    board = [random.randint(0, n - 1) for _ in range(n)]  # Random initial placement
    cols = [0] * n
    plus_diags = [0] * (2 * n - 1)
    minus_diags = [0] * (2 * n - 1)
    conflicts = 0

    # Initialize conflicts
    for row in range(n):
        col = board[row]
        conflicts += cols[col] + plus_diags[row + col] + minus_diags[row - col + n - 1]
        cols[col] += 1
        plus_diags[row + col] += 1
        minus_diags[row - col + n - 1] += 1

    return board, cols, plus_diags, minus_diags, conflicts

# ====================================================================================================
def calculate_delta(row, new_col, board, cols, plus_diags, minus_diags, n):
    old_col = board[row]
    delta = 0

    # Remove current conflicts (before moving the queen)
    delta -= (cols[old_col] - 1) + (plus_diags[row + old_col] - 1) + (minus_diags[row - old_col + n - 1] - 1)

    # Add new conflicts (after moving the queen)
    delta += (cols[new_col]) + (plus_diags[row + new_col]) + (minus_diags[row - new_col + n - 1])

    return delta

# ====================================================================================================
def move_queen(row, new_col, board, cols, plus_diags, minus_diags, n):
    old_col = board[row]

    # Remove the queen from the current position
    cols[old_col] -= 1
    plus_diags[row + old_col] -= 1
    minus_diags[row - old_col + n - 1] -= 1

    # Place the queen in the new position
    cols[new_col] += 1
    plus_diags[row + new_col] += 1
    minus_diags[row - new_col + n - 1] += 1

    # Update the board
    board[row] = new_col

# ====================================================================================================
def validate_board(board, cols, plus_diags, minus_diags, n):
    for value in cols + plus_diags + minus_diags:
        if value > 1:
            return False
    return True

# ====================================================================================================
def calculate_total_conflicts(board, n):
    cols = [0] * n
    plus_diags = [0] * (2 * n - 1)
    minus_diags = [0] * (2 * n - 1)
    conflicts = 0

    for row in range(n):
        col = board[row]
        cols[col] += 1
        plus_diags[row + col] += 1
        minus_diags[row - col + n - 1] += 1

    for value in cols + plus_diags + minus_diags:
        if value > 1:
            conflicts += value - 1

    return conflicts

# ====================================================================================================
# Hill climbing algorithm with simulated annealing
def hill_climbing_with_optimization(n, max_steps, temperature=1.0, cooling_rate=0.999):
    board, cols, plus_diags, minus_diags, conflicts = initialize_board(n)

    for step in range(max_steps):
        if conflicts == 0:
            return board, step

        # Choose a random row to move a queen
        row = random.randint(0, n - 1)
        new_col = random.randint(0, n - 1)
        delta = calculate_delta(row, new_col, board, cols, plus_diags, minus_diags, n)
        
        if  delta <= 0:
            move_queen(row, new_col, board, cols, plus_diags, minus_diags, n)
            conflicts += delta
            
        elif random.uniform(0, 1) < np.exp(-delta / temperature):
            move_queen(row, new_col, board, cols, plus_diags, minus_diags, n)
            conflicts += delta
            
        temperature *= cooling_rate

    return board, max_steps  # No solution found

#  ====================================================================================================
def print_board(board):
    """Prints the board in a readable chessboard format."""
    n = len(board)
    for row in range(n):
        line = ['Q' if col == board[row] else 'x' for col in range(n)]
        print(' '.join(line))
    print()


# Run the algorithm
n = 10 # Board size
max_steps = 1000  # Maximum number of steps

solution, steps = hill_climbing_with_optimization(n, max_steps)
if calculate_total_conflicts(solution, n) == 0:
    print(f"Solution found after {steps} steps:")
    print_board(solution)
    
else:
    print(f"No solution found after {max_steps} steps.")
