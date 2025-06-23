import math
import random

class Puzzle15Env:
    def __init__(self, shuffle_steps=1000):
        self.board = list(range(1, 16)) + [0]
        self.blank_pos = 15
        self.phi = 0
        self._shuffle(shuffle_steps)
        self._calculate_heuristic()
# ==============================================================================
    def _shuffle(self, steps):
        for _ in range(steps):
            moves = self.get_valid_moves()
            if not moves:
                break
            self._swap_blank(random.choice(moves))
# ==============================================================================
    def _swap_blank(self, new_pos):
        self.board[self.blank_pos], self.board[new_pos] = self.board[new_pos], 0
        self.blank_pos = new_pos
# ==============================================================================
    def get_valid_moves(self):
        moves = []
        row, col = self.blank_pos // 4, self.blank_pos % 4
        if row > 0: moves.append(self.blank_pos - 4)
        if row < 3: moves.append(self.blank_pos + 4)
        if col > 0: moves.append(self.blank_pos - 1)
        if col < 3: moves.append(self.blank_pos + 1)
        return moves
# ==============================================================================
    def _calculate_heuristic(self):
        self.phi = self._manhattan_linear_conflict()


    def _manhattan_linear_conflict(self):
        manhattan = 0
        conflicts = 0
        
        # Row conflicts
        for row in range(4):
            tiles = []
            for col in range(4):
                i = row * 4 + col
                tile = self.board[i]
                if tile == 0: continue
                tiles.append(tile)
                target_row, target_col = (tile - 1) // 4, (tile - 1) % 4
                manhattan += abs(row - target_row) + abs(col - target_col)
                
                if target_row == row:
                    for t in tiles:
                        if t != tile and (t - 1) // 4 == row and tile < t:
                            conflicts += 2
        
        # Column conflicts
        for col in range(4):
            tiles = []
            for row in range(4):
                i = row * 4 + col
                tile = self.board[i]
                if tile == 0: continue
                target_row, target_col = (tile - 1) // 4, (tile - 1) % 4
                if target_col == col:
                    for t in tiles:
                        if t != tile and (t - 1) % 4 == col and tile < t:
                            conflicts += 2
                tiles.append(tile)
        
        return manhattan + conflicts
# ==============================================================================
    def make_move(self, tile_pos):
        if tile_pos not in self.get_valid_moves():
            return None, 0

        move_info = (tile_pos, self.blank_pos, self.board[tile_pos])
        old_phi = self.phi
        
        self.board[self.blank_pos], self.board[tile_pos] = self.board[tile_pos], 0
        self.blank_pos = tile_pos
        
        self._calculate_heuristic()
        
        return move_info, self.phi - old_phi
# ==============================================================================
    def unmake_move(self, move_info):
        tile_pos, blank_pos, tile = move_info
        self.board[blank_pos], self.board[tile_pos] = 0, tile
        self.blank_pos = blank_pos
        self._calculate_heuristic()
# ==============================================================================
    def display(self):
        return "\n".join(" ".join(f"{tile:2}" if tile != 0 else "  "
                         for tile in self.board[i*4:(i+1)*4])
                        for i in range(4))