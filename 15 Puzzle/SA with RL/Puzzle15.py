import math
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
class Puzzle15Env:
    def __init__(self, shuffle_steps=1000, use_learned_weights=False):
        self.board = list(range(1, 16)) + [0]
        self.blank_pos = 15
        self.phi = 0
        self.use_learned_weights = use_learned_weights
        
        # Initialize weights for each tile position (4x4 grid)
        # Start with Manhattan distance baseline weights
        self.weights = np.ones((4, 4))
        
        # Cache initial state for feature extraction
        self.initial_board = None
        
        if shuffle_steps > 0:
            self._shuffle(shuffle_steps)
            self.initial_board = self.board.copy()
        self._calculate_heuristic()
        
    def set_weights(self, weights):
        """Update the learned weights"""
        self.weights = weights.reshape(4, 4)
        self._calculate_heuristic()
        
    def get_features(self):
        """Extract Manhattan distances as features for each tile"""
        features = np.zeros((4, 4))
        
        for pos in range(16):
            tile = self.board[pos]
            if tile == 0:
                continue
                
            # Current position
            curr_row, curr_col = pos // 4, pos % 4
            
            # Target position for this tile
            target_pos = tile - 1
            target_row, target_col = target_pos // 4, target_pos % 4
            
            # Manhattan distance
            manhattan_dist = abs(curr_row - target_row) + abs(curr_col - target_col)
            
            # Store feature at the target position
            features[target_row, target_col] = manhattan_dist
            
        return features
    
    def _calculate_heuristic(self):
        """Calculate weighted Manhattan distance"""
        if self.use_learned_weights:
            # Use learned weights
            features = self.get_features()
            self.phi = np.sum(self.weights * features)
        else:
            # Use standard Manhattan distance with linear conflicts
            self.phi = self._manhattan_linear_conflict()
    
    def _shuffle(self, steps):
        """Shuffle the puzzle from solved state"""
        for _ in range(steps):
            moves = self.get_valid_moves()
            if not moves:
                break
            self._swap_blank(random.choice(moves))
    
    def _swap_blank(self, new_pos):
        self.board[self.blank_pos], self.board[new_pos] = self.board[new_pos], 0
        self.blank_pos = new_pos
    
    def get_valid_moves(self):
        moves = []
        row, col = self.blank_pos // 4, self.blank_pos % 4
        if row > 0: moves.append(self.blank_pos - 4)
        if row < 3: moves.append(self.blank_pos + 4)
        if col > 0: moves.append(self.blank_pos - 1)
        if col < 3: moves.append(self.blank_pos + 1)
        return moves
    
    def _manhattan_linear_conflict(self):
        """Original heuristic with Manhattan distance and linear conflicts"""
        manhattan = 0
        conflicts = 0
        
        for pos in range(16):
            tile = self.board[pos]
            if tile == 0:
                continue
                
            curr_row, curr_col = pos // 4, pos % 4
            target_row, target_col = (tile - 1) // 4, (tile - 1) % 4
            manhattan += abs(curr_row - target_row) + abs(curr_col - target_col)
        
        # Add linear conflicts
        # Row conflicts
        for row in range(4):
            tiles_in_row = []
            for col in range(4):
                pos = row * 4 + col
                tile = self.board[pos]
                if tile != 0 and (tile - 1) // 4 == row:
                    tiles_in_row.append((tile, col))
            
            # Check for conflicts
            for i in range(len(tiles_in_row)):
                for j in range(i + 1, len(tiles_in_row)):
                    tile1, col1 = tiles_in_row[i]
                    tile2, col2 = tiles_in_row[j]
                    target_col1 = (tile1 - 1) % 4
                    target_col2 = (tile2 - 1) % 4
                    
                    if col1 < col2 and target_col1 > target_col2:
                        conflicts += 2
        
        # Column conflicts
        for col in range(4):
            tiles_in_col = []
            for row in range(4):
                pos = row * 4 + col
                tile = self.board[pos]
                if tile != 0 and (tile - 1) % 4 == col:
                    tiles_in_col.append((tile, row))
            
            # Check for conflicts
            for i in range(len(tiles_in_col)):
                for j in range(i + 1, len(tiles_in_col)):
                    tile1, row1 = tiles_in_col[i]
                    tile2, row2 = tiles_in_col[j]
                    target_row1 = (tile1 - 1) // 4
                    target_row2 = (tile2 - 1) // 4
                    
                    if row1 < row2 and target_row1 > target_row2:
                        conflicts += 2
        
        return manhattan + conflicts
    
    def make_move(self, tile_pos):
        if tile_pos not in self.get_valid_moves():
            return None, 0

        move_info = (tile_pos, self.blank_pos, self.board[tile_pos])
        old_phi = self.phi
        
        self.board[self.blank_pos], self.board[tile_pos] = self.board[tile_pos], 0
        self.blank_pos = tile_pos
        
        self._calculate_heuristic()
        
        return move_info, self.phi - old_phi
    
    def unmake_move(self, move_info):
        tile_pos, blank_pos, tile = move_info
        self.board[blank_pos], self.board[tile_pos] = 0, tile
        self.blank_pos = blank_pos
        self._calculate_heuristic()
    
    def is_solved(self):
        return self.board == list(range(1, 16)) + [0]
    
    def display(self):
        return "\n".join(" ".join(f"{tile:2}" if tile != 0 else "  "
                         for tile in self.board[i*4:(i+1)*4])
                        for i in range(4))