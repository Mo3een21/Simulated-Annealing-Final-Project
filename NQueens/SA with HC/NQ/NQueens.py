import random
import math

def initialize_board(n):
    """Creates an initial board where each queen is placed in a random column in its row."""
    return [random.randint(0, n - 1) for _ in range(n)]

def calculate_conflicts(board):
    # change the function to be O(n) instead of O(n^2) using a dictionary
    """Calculates the number of conflicts between queens on the board."""
    n = len(board)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                #add condition to check if the queens are in the same diagonal ****
                conflicts += 1
    return conflicts

def get_neighbors(board):
    """Generates all possible neighbor boards."""
    n = len(board)
    neighbors = []
    for row in range(n):
        for col in range(n):
            if col != board[row]:
                neighbor = board[:]
                neighbor[row] = col
                neighbors.append(neighbor)
    return neighbors

def print_board(board):
    """Prints the board in a readable chessboard format."""
    n = len(board)
    for row in range(n):
        line = ['Q' if col == board[row] else 'x' for col in range(n)]
        print(' '.join(line))
    print()  # Add a blank line for better readability

def hill_climbing_simulated_annealing(n, max_steps, initial_temperature, cooling_rate):
    """Solves the N-Queens problem using Hill Climbing combined with Simulated Annealing."""
    current_board = initialize_board(n)
    current_conflicts = calculate_conflicts(current_board)
    temperature = initial_temperature

    for step in range(max_steps):
        if current_conflicts == 0:
            return current_board, step  # A valid board was found

        # Get neighbors
        neighbors = get_neighbors(current_board)
        next_board = random.choice(neighbors)
        next_conflicts = calculate_conflicts(next_board)

        # Calculate the probability of moving to the next state
        delta = next_conflicts - current_conflicts
        if delta < 0  or random.random() < math.exp(-delta / temperature):
            current_board = next_board
            current_conflicts = next_conflicts

        # Cool down the temperature
        temperature *= cooling_rate

    return current_board, max_steps  # If no solution was found

# Define parameters and run the algorithm
n =12  # Board size (e.g., 8 queens)
max_steps = 1000  # Maximum number of steps
initial_temperature = 100  # Initial temperature
cooling_rate = 0.99  # Cooling rate

solution, steps = hill_climbing_simulated_annealing(n, max_steps, initial_temperature, cooling_rate)
if calculate_conflicts(solution) == 0:
    print(f"Solution found after {steps} steps: {solution}")
    print_board(solution)  # Print the whole board
else:
    print(f"No solution found after {max_steps} steps.{solution} Final board:")
    print_board(solution)  # Print the final board