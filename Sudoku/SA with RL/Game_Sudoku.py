import math
import random


class SudokuEnv:
    def __init__(self, initial_grid):
        
        self.board = [row.copy() for row in initial_grid]
        self.fixed = [[cell != 0 for cell in row] for row in initial_grid]

        self.row_counts = [[0] * 10 for _ in range(9)]
        self.col_counts = [[0] * 10 for _ in range(9)]

        for subgrid_row in range(3):
            for subgrid_col in range(3):
                self._initialize_subgrid(subgrid_row, subgrid_col)

        for r in range(9):
            for c in range(9):
                val = self.board[r][c]
                self.row_counts[r][val] += 1
                self.col_counts[c][val] += 1

        self._calculate_initial_heuristic()

        self.cell_conflicts = [[0] * 9 for _ in range(9)]
        self._calculate_all_cell_conflicts()
# =====================================================================
# this function initializes a subgrid by filling it with numbers 1-9
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
# ====================================================================
# this function calculates the initial heuristic value (phi) based on row and column conflicts
    def _calculate_initial_heuristic(self):
        row_conflicts = 0
        col_conflicts = 0
        for r in range(9):
            for v in range(1, 10):
                if self.row_counts[r][v] > 1:
                    row_conflicts += self.row_counts[r][v] - 1
        for c in range(9):
            for v in range(1, 10):
                if self.col_counts[c][v] > 1:
                    col_conflicts += self.col_counts[c][v] - 1

        self.phi = row_conflicts + col_conflicts
# ====================================================================
# this function calculates the conflicts for all cells in the Sudoku board
    def _calculate_all_cell_conflicts(self):
        for r in range(9):
            for c in range(9):
                val = self.board[r][c]
                if self.fixed[r][c]:
                    self.cell_conflicts[r][c] = 0
                else:
                    # כמות החזרות של val בשורה r ועמודה c
                    rc = max(0, self.row_counts[r][val] - 1)
                    cc = max(0, self.col_counts[c][val] - 1)
                    self.cell_conflicts[r][c] = rc + cc
# ====================================================================
# this function makes a move by choosing a subgrid and swapping two cells within it
    def make_move(self, chosen_subgrid=None):
        self._calculate_all_cell_conflicts()

        if chosen_subgrid is None:
            subgrid_row = random.randint(0, 2)
            subgrid_col = random.randint(0, 2)
        else:
            subgrid_row, subgrid_col = chosen_subgrid

        start_r = subgrid_row * 3
        start_c = subgrid_col * 3

        cells = []
        for i in range(start_r, start_r + 3):
            for j in range(start_c, start_c + 3):
                if not self.fixed[i][j]:
                    cells.append((i, j))

        if len(cells) < 2:
            return None, 0

        weights = []
        for (i, j) in cells:
            weights.append(1 + self.cell_conflicts[i][j])
# ===================================================================
# this function randomly selects two cells from the subgrid and swaps them
        def weighted_choice(choices, weights_list):
            total = sum(weights_list)
            r = random.uniform(0, total)
            upto = 0
            for idx, w in enumerate(weights_list):
                if upto + w >= r:
                    return choices[idx]
                upto += w
            return choices[-1]

        a = weighted_choice(cells, weights)
        idx_a = cells.index(a)
        cells.pop(idx_a)
        weights.pop(idx_a)

        b = weighted_choice(cells, weights)

        return self._swap_cells(a, b)
# ===================================================================
# this function swaps two cells and calculates the change in heuristic value (delta)
    def _swap_cells(self, pos1, pos2):
        (r1, c1), (r2, c2) = pos1, pos2
        val1 = self.board[r1][c1]
        val2 = self.board[r2][c2]

        old_phi = self.phi

        self._decrement_count(r1, c1, val1)
        self._decrement_count(r2, c2, val2)

        self._increment_count(r1, c1, val2)
        self._increment_count(r2, c2, val1)

        self.board[r1][c1], self.board[r2][c2] = val2, val1

        delta = self.phi - old_phi
        return (pos1, pos2), delta
# ===================================================================
# this function undoes a move by swapping the cells back to their original positions
    def unmake_move(self, move_info):
        pos1, pos2 = move_info
        self._swap_cells(pos1, pos2)
# ====================================================================
# this function returns the current heuristic value (phi)
    def _decrement_count(self, r, c, val):
        cnt_row = self.row_counts[r][val]
        if cnt_row > 1:
            self.phi -= 1
        self.row_counts[r][val] -= 1

        cnt_col = self.col_counts[c][val]
        if cnt_col > 1:
            self.phi -= 1
        self.col_counts[c][val] -= 1
# ===================================================================
# this function increments the count of a value in a row and column, updating the heuristic value
    def _increment_count(self, r, c, val):
        cnt_row = self.row_counts[r][val]
        if cnt_row >= 1:
            self.phi += 1
        self.row_counts[r][val] += 1

        cnt_col = self.col_counts[c][val]
        if cnt_col >= 1:
            self.phi += 1
        self.col_counts[c][val] += 1
# ====================================================================
# draws the Sudoku board in a readable format
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

# ====================================================================
# generates a Sudoku puzzle with a specified difficulty level
def generate_sudoku_puzzle(difficulty=40):
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

    digits = list(range(1, 10))
    random.shuffle(digits)
    for i in range(9):
        for j in range(9):
            base[i][j] = digits[base[i][j] - 1]

    for band in range(0, 9, 3):
        rows = list(range(band, band + 3))
        random.shuffle(rows)
        base[band:band + 3] = [base[r] for r in rows]

    for stack in range(0, 9, 3):
        cols = list(range(stack, stack + 3))
        random.shuffle(cols)
        for row in base:
            row[stack:stack + 3] = [row[c] for c in cols]

    puzzle = [row.copy() for row in base]
    empty_cells = set()
    while len(empty_cells) < difficulty:
        i = random.randint(0, 8)
        j = random.randint(0, 8)
        if (i, j) not in empty_cells:
            puzzle[i][j] = 0
            empty_cells.add((i, j))

    return puzzle