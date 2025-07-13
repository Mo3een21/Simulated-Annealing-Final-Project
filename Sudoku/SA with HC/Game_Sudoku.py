import math
import random

class SudokuEnv:
    def __init__(self, initial_grid):
        self.board = [row.copy() for row in initial_grid]
        self.fixed = [[cell != 0 for cell in row] for row in initial_grid]
        
        # Initialize with valid subgrids
        for subgrid_row in range(3):
            for subgrid_col in range(3):
                self._initialize_subgrid(subgrid_row, subgrid_col)
        
        self._calculate_heuristic()

    def _initialize_subgrid(self, subgrid_row, subgrid_col):
        start_r = subgrid_row * 3
        start_c = subgrid_col * 3
        present = []
        empty = []
        for i in range(start_r, start_r + 3):
            for j in range(start_c, start_c + 3):
                num = self.board[i][j]
                if num != 0:
                    present.append(num)
                else:
                    empty.append((i, j))
        missing = list(set(range(1, 10)) - set(present))
        random.shuffle(missing)
        for (i, j), num in zip(empty, missing):
            self.board[i][j] = num

    def _calculate_heuristic(self):
        # ספירת קונפליקטים בשורות ועמודות
        row_conflicts = sum(len(row) - len(set(row)) for row in self.board)
        col_conflicts = sum(len(col) - len(set(col)) for col in zip(*self.board))
        # ספירת קונפליקטים בכל בלוק 3x3
        block_conflicts = 0
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                block = [self.board[r][c] for r in range(i, i+3) for c in range(j, j+3)]
                block_conflicts += len(block) - len(set(block))
        self.phi = row_conflicts + col_conflicts + block_conflicts


    def make_move(self):
        # Find a random subgrid with swappable cells
        subgrid_row = random.randint(0, 2)
        subgrid_col = random.randint(0, 2)
        start_r = subgrid_row * 3
        start_c = subgrid_col * 3
        
        # Collect non-fixed cells in subgrid
        cells = []
        for i in range(start_r, start_r + 3):
            for j in range(start_c, start_c + 3):
                if not self.fixed[i][j]:
                    cells.append((i, j))
        
        if len(cells) < 2:
            return None, 0
        
        a, b = random.sample(cells, 2)
        return self._swap_cells(a, b)

    def _swap_cells(self, pos1, pos2):
        # Swap cells and calculate delta cost
        old_phi = self.phi
        self.board[pos1[0]][pos1[1]], self.board[pos2[0]][pos2[1]] = \
            self.board[pos2[0]][pos2[1]], self.board[pos1[0]][pos1[1]]
        
        self._calculate_heuristic()
        delta = self.phi - old_phi
        return (pos1, pos2), delta

    def unmake_move(self, move_info):
        # Swap back
        pos1, pos2 = move_info
        self._swap_cells(pos1, pos2)

    def display(self):
        output = []
        for i, row in enumerate(self.board):
            if i % 3 == 0 and i != 0:
                output.append("-" * 21)
            row_str = []
            for j, num in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str.append("|")
                row_str.append(str(num) if num != 0 else " ")
            output.append(" ".join(row_str))
        return "\n".join(output)

def generate_sudoku_puzzle(difficulty=40):
    """Generate a random Sudoku puzzle with given number of empty cells"""
    # Start with a valid solved grid
    base = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]
    
    # Random permutations to create new puzzle
    # Shuffle numbers
    digits = list(range(1, 10))
    random.shuffle(digits)
    for i in range(9):
        for j in range(9):
            base[i][j] = digits[base[i][j]-1] if base[i][j] != 0 else 0
    
    # Shuffle rows within bands
    for band in range(0, 9, 3):
        rows = list(range(band, band+3))
        random.shuffle(rows)
        base[band:band+3] = [base[r] for r in rows]
    
    # Shuffle columns within stacks
    for stack in range(0, 9, 3):
        cols = list(range(stack, stack+3))
        random.shuffle(cols)
        for row in base: 
            row[stack:stack+3] = [row[c] for c in cols]
    
    # Remove random cells
    puzzle = [row.copy() for row in base]
    empty_cells = set()
    while len(empty_cells) < difficulty:
        i = random.randint(0, 8)
        j = random.randint(0, 8)
        if (i, j) not in empty_cells:
            puzzle[i][j] = 0
            empty_cells.add((i, j))
    
    return puzzle
