import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle


# ==============================================================================
# Enhanced Rubik's Cube Environment with State Representation
# ==============================================================================
class RubiksCubeEnv:
    def __init__(self, shuffle_steps=100):
        # Initialize solved cube state - each face has 9 stickers (3x3)
        # 0=White, 1=Red, 2=Blue, 3=Orange, 4=Green, 5=Yellow
        self.cube = {
            'U': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Up (White)
            'R': [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Right (Red)  
            'F': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # Front (Blue)
            'D': [[5, 5, 5], [5, 5, 5], [5, 5, 5]],  # Down (Yellow)
            'L': [[4, 4, 4], [4, 4, 4], [4, 4, 4]],  # Left (Green)
            'B': [[3, 3, 3], [3, 3, 3], [3, 3, 3]]   # Back (Orange)
        }
        
        # Standard Rubik's cube moves
        self.moves = ['F', "F'", 'R', "R'", 'U', "U'", 'L', "L'", 'D', "D'", 'B', "B'"]
        self.move_to_idx = {move: idx for idx, move in enumerate(self.moves)}
        self.phi = 0
        self.shuffle_history = []
        
        if shuffle_steps > 0:
            self._shuffle(shuffle_steps)
        self._calculate_heuristic()
        
    def get_state_vector(self):
        """Convert cube state to a neural network input vector"""
        # Flatten the cube state into a 1D vector
        # Each face has 9 stickers, 6 faces = 54 values
        state = []
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            for row in self.cube[face]:
                for sticker in row:
                    # One-hot encode each sticker color
                    one_hot = [0] * 6
                    one_hot[sticker] = 1
                    state.extend(one_hot)
        
        # Add additional features
        # Corner and edge correctness indicators
        corner_features = self._get_corner_features()
        edge_features = self._get_edge_features()
        
        state.extend(corner_features)
        state.extend(edge_features)
        
        # Add current heuristic value (normalized)
        state.append(self.phi / 100.0)
        
        return np.array(state, dtype=np.float32)
    
    def _get_corner_features(self):
        """Extract features about corner piece positions"""
        features = []
        corner_positions = [
            [('U',0,0), ('L',0,0), ('B',0,2)], [('U',0,2), ('B',0,0), ('R',0,2)],
            [('U',2,0), ('F',0,0), ('L',0,2)], [('U',2,2), ('R',0,0), ('F',0,2)],
            [('D',0,0), ('L',2,2), ('F',2,0)], [('D',0,2), ('F',2,2), ('R',2,0)],
            [('D',2,0), ('B',2,0), ('L',2,0)], [('D',2,2), ('R',2,2), ('B',2,2)]
        ]
        
        target_corners = [
            [0,4,3], [0,3,1], [0,2,4], [0,1,2],
            [5,4,2], [5,2,1], [5,3,4], [5,1,3]
        ]
        
        for i, corner_pos in enumerate(corner_positions):
            current = [self.cube[face][row][col] for face, row, col in corner_pos]
            target = target_corners[i]
            
            # Is corner in correct position?
            features.append(1.0 if set(current) == set(target) else 0.0)
            # Is corner correctly oriented?
            features.append(1.0 if current == target else 0.0)
            
        return features
    
    def _get_edge_features(self):
        """Extract features about edge piece positions"""
        features = []
        edge_positions = [
            [('U',0,1), ('B',0,1)], [('U',1,2), ('R',0,1)], 
            [('U',2,1), ('F',0,1)], [('U',1,0), ('L',0,1)],
            [('F',1,0), ('L',1,2)], [('F',1,2), ('R',1,0)],
            [('B',1,0), ('R',1,2)], [('B',1,2), ('L',1,0)],
            [('D',0,1), ('F',2,1)], [('D',1,2), ('R',2,1)],
            [('D',2,1), ('B',2,1)], [('D',1,0), ('L',2,1)]
        ]
        
        target_edges = [
            [0,3], [0,1], [0,2], [0,4],
            [2,4], [2,1], [3,1], [3,4],
            [5,2], [5,1], [5,3], [5,4]
        ]
        
        for i, edge_pos in enumerate(edge_positions):
            current = [self.cube[face][row][col] for face, row, col in edge_pos]
            target = target_edges[i]
            
            # Is edge in correct position?
            features.append(1.0 if set(current) == set(target) else 0.0)
            # Is edge correctly oriented?
            features.append(1.0 if current == target else 0.0)
            
        return features

    def _shuffle(self, steps):
        """Shuffle cube and keep track of moves for curriculum learning"""
        self.shuffle_history = []
        for _ in range(steps):
            move = random.choice(self.moves)
            self._execute_move(move)
            self.shuffle_history.append(move)
    
    def is_solved(self):
        """Check if cube is in solved state"""
        return self.phi == 0
    
    def get_reward(self, old_phi):
        """Calculate reward based on heuristic change"""
        if self.is_solved():
            return 100.0  # Large reward for solving
        
        # Reward for improving heuristic
        delta = old_phi - self.phi
        if delta > 0:
            return delta * 2.0  # Positive reward for improvement
        else:
            return delta * 0.5  # Smaller penalty for worsening
    
    # [Previous move execution methods remain the same]
    def _execute_move(self, move):
        """Execute a single move on the cube"""
        if move == 'F':
            self._rotate_face('F')
            temp = [self.cube['U'][2][0], self.cube['U'][2][1], self.cube['U'][2][2]]
            self.cube['U'][2][0], self.cube['U'][2][1], self.cube['U'][2][2] = self.cube['L'][2][2], self.cube['L'][1][2], self.cube['L'][0][2]
            self.cube['L'][0][2], self.cube['L'][1][2], self.cube['L'][2][2] = self.cube['D'][0][2], self.cube['D'][0][1], self.cube['D'][0][0]
            self.cube['D'][0][0], self.cube['D'][0][1], self.cube['D'][0][2] = self.cube['R'][2][0], self.cube['R'][1][0], self.cube['R'][0][0]
            self.cube['R'][0][0], self.cube['R'][1][0], self.cube['R'][2][0] = temp[0], temp[1], temp[2]
            
        elif move == "F'":
            self._rotate_face_prime('F')  
            temp = [self.cube['U'][2][0], self.cube['U'][2][1], self.cube['U'][2][2]]
            self.cube['U'][2][0], self.cube['U'][2][1], self.cube['U'][2][2] = self.cube['R'][0][0], self.cube['R'][1][0], self.cube['R'][2][0]
            self.cube['R'][0][0], self.cube['R'][1][0], self.cube['R'][2][0] = self.cube['D'][0][2], self.cube['D'][0][1], self.cube['D'][0][0]
            self.cube['D'][0][0], self.cube['D'][0][1], self.cube['D'][0][2] = self.cube['L'][2][2], self.cube['L'][1][2], self.cube['L'][0][2]
            self.cube['L'][0][2], self.cube['L'][1][2], self.cube['L'][2][2] = temp[2], temp[1], temp[0]
            
        elif move == 'R':
            self._rotate_face('R')
            temp = [self.cube['U'][0][2], self.cube['U'][1][2], self.cube['U'][2][2]]
            self.cube['U'][0][2], self.cube['U'][1][2], self.cube['U'][2][2] = self.cube['F'][0][2], self.cube['F'][1][2], self.cube['F'][2][2]
            self.cube['F'][0][2], self.cube['F'][1][2], self.cube['F'][2][2] = self.cube['D'][0][2], self.cube['D'][1][2], self.cube['D'][2][2]
            self.cube['D'][0][2], self.cube['D'][1][2], self.cube['D'][2][2] = self.cube['B'][2][0], self.cube['B'][1][0], self.cube['B'][0][0]
            self.cube['B'][0][0], self.cube['B'][1][0], self.cube['B'][2][0] = temp[2], temp[1], temp[0]
            
        elif move == "R'":
            self._rotate_face_prime('R')
            temp = [self.cube['U'][0][2], self.cube['U'][1][2], self.cube['U'][2][2]]
            self.cube['U'][0][2], self.cube['U'][1][2], self.cube['U'][2][2] = self.cube['B'][2][0], self.cube['B'][1][0], self.cube['B'][0][0]
            self.cube['B'][0][0], self.cube['B'][1][0], self.cube['B'][2][0] = self.cube['D'][2][2], self.cube['D'][1][2], self.cube['D'][0][2]
            self.cube['D'][0][2], self.cube['D'][1][2], self.cube['D'][2][2] = self.cube['F'][0][2], self.cube['F'][1][2], self.cube['F'][2][2]
            self.cube['F'][0][2], self.cube['F'][1][2], self.cube['F'][2][2] = temp[0], temp[1], temp[2]
            
        elif move == 'U':
            self._rotate_face('U')
            temp = [self.cube['F'][0][0], self.cube['F'][0][1], self.cube['F'][0][2]]
            self.cube['F'][0][0], self.cube['F'][0][1], self.cube['F'][0][2] = self.cube['R'][0][0], self.cube['R'][0][1], self.cube['R'][0][2]
            self.cube['R'][0][0], self.cube['R'][0][1], self.cube['R'][0][2] = self.cube['B'][0][0], self.cube['B'][0][1], self.cube['B'][0][2]
            self.cube['B'][0][0], self.cube['B'][0][1], self.cube['B'][0][2] = self.cube['L'][0][0], self.cube['L'][0][1], self.cube['L'][0][2]
            self.cube['L'][0][0], self.cube['L'][0][1], self.cube['L'][0][2] = temp[0], temp[1], temp[2]
            
        elif move == "U'":
            self._rotate_face_prime('U')
            temp = [self.cube['F'][0][0], self.cube['F'][0][1], self.cube['F'][0][2]]
            self.cube['F'][0][0], self.cube['F'][0][1], self.cube['F'][0][2] = self.cube['L'][0][0], self.cube['L'][0][1], self.cube['L'][0][2]
            self.cube['L'][0][0], self.cube['L'][0][1], self.cube['L'][0][2] = self.cube['B'][0][0], self.cube['B'][0][1], self.cube['B'][0][2]
            self.cube['B'][0][0], self.cube['B'][0][1], self.cube['B'][0][2] = self.cube['R'][0][0], self.cube['R'][0][1], self.cube['R'][0][2]
            self.cube['R'][0][0], self.cube['R'][0][1], self.cube['R'][0][2] = temp[0], temp[1], temp[2]
            
        elif move == 'L':
            self._rotate_face('L')
            temp = [self.cube['U'][0][0], self.cube['U'][1][0], self.cube['U'][2][0]]
            self.cube['U'][0][0], self.cube['U'][1][0], self.cube['U'][2][0] = self.cube['B'][2][2], self.cube['B'][1][2], self.cube['B'][0][2]
            self.cube['B'][0][2], self.cube['B'][1][2], self.cube['B'][2][2] = self.cube['D'][2][0], self.cube['D'][1][0], self.cube['D'][0][0]
            self.cube['D'][0][0], self.cube['D'][1][0], self.cube['D'][2][0] = self.cube['F'][0][0], self.cube['F'][1][0], self.cube['F'][2][0]
            self.cube['F'][0][0], self.cube['F'][1][0], self.cube['F'][2][0] = temp[0], temp[1], temp[2]
            
        elif move == "L'":
            self._rotate_face_prime('L')
            temp = [self.cube['U'][0][0], self.cube['U'][1][0], self.cube['U'][2][0]]
            self.cube['U'][0][0], self.cube['U'][1][0], self.cube['U'][2][0] = self.cube['F'][0][0], self.cube['F'][1][0], self.cube['F'][2][0]
            self.cube['F'][0][0], self.cube['F'][1][0], self.cube['F'][2][0] = self.cube['D'][0][0], self.cube['D'][1][0], self.cube['D'][2][0]
            self.cube['D'][0][0], self.cube['D'][1][0], self.cube['D'][2][0] = self.cube['B'][2][2], self.cube['B'][1][2], self.cube['B'][0][2]
            self.cube['B'][0][2], self.cube['B'][1][2], self.cube['B'][2][2] = temp[2], temp[1], temp[0]
            
        elif move == 'D':
            self._rotate_face('D')
            temp = [self.cube['F'][2][0], self.cube['F'][2][1], self.cube['F'][2][2]]
            self.cube['F'][2][0], self.cube['F'][2][1], self.cube['F'][2][2] = self.cube['L'][2][0], self.cube['L'][2][1], self.cube['L'][2][2]
            self.cube['L'][2][0], self.cube['L'][2][1], self.cube['L'][2][2] = self.cube['B'][2][0], self.cube['B'][2][1], self.cube['B'][2][2]
            self.cube['B'][2][0], self.cube['B'][2][1], self.cube['B'][2][2] = self.cube['R'][2][0], self.cube['R'][2][1], self.cube['R'][2][2]
            self.cube['R'][2][0], self.cube['R'][2][1], self.cube['R'][2][2] = temp[0], temp[1], temp[2]
            
        elif move == "D'":
            self._rotate_face_prime('D')
            temp = [self.cube['F'][2][0], self.cube['F'][2][1], self.cube['F'][2][2]]
            self.cube['F'][2][0], self.cube['F'][2][1], self.cube['F'][2][2] = self.cube['R'][2][0], self.cube['R'][2][1], self.cube['R'][2][2]
            self.cube['R'][2][0], self.cube['R'][2][1], self.cube['R'][2][2] = self.cube['B'][2][0], self.cube['B'][2][1], self.cube['B'][2][2]
            self.cube['B'][2][0], self.cube['B'][2][1], self.cube['B'][2][2] = self.cube['L'][2][0], self.cube['L'][2][1], self.cube['L'][2][2]
            self.cube['L'][2][0], self.cube['L'][2][1], self.cube['L'][2][2] = temp[0], temp[1], temp[2]
            
        elif move == 'B':
            self._rotate_face('B')
            temp = [self.cube['U'][0][0], self.cube['U'][0][1], self.cube['U'][0][2]]
            self.cube['U'][0][0], self.cube['U'][0][1], self.cube['U'][0][2] = self.cube['R'][0][2], self.cube['R'][1][2], self.cube['R'][2][2]
            self.cube['R'][0][2], self.cube['R'][1][2], self.cube['R'][2][2] = self.cube['D'][2][2], self.cube['D'][2][1], self.cube['D'][2][0]
            self.cube['D'][2][0], self.cube['D'][2][1], self.cube['D'][2][2] = self.cube['L'][2][0], self.cube['L'][1][0], self.cube['L'][0][0]
            self.cube['L'][0][0], self.cube['L'][1][0], self.cube['L'][2][0] = temp[2], temp[1], temp[0]
            
        elif move == "B'":
            self._rotate_face_prime('B')
            temp = [self.cube['U'][0][0], self.cube['U'][0][1], self.cube['U'][0][2]]
            self.cube['U'][0][0], self.cube['U'][0][1], self.cube['U'][0][2] = self.cube['L'][2][0], self.cube['L'][1][0], self.cube['L'][0][0]
            self.cube['L'][0][0], self.cube['L'][1][0], self.cube['L'][2][0] = self.cube['D'][2][2], self.cube['D'][2][1], self.cube['D'][2][0]
            self.cube['D'][2][0], self.cube['D'][2][1], self.cube['D'][2][2] = self.cube['R'][2][2], self.cube['R'][1][2], self.cube['R'][0][2]
            self.cube['R'][0][2], self.cube['R'][1][2], self.cube['R'][2][2] = temp[2], temp[1], temp[0]

    def _rotate_face(self, face):
        """Rotate a face 90 degrees clockwise"""
        old = [row[:] for row in self.cube[face]]
        self.cube[face][0][0] = old[2][0]
        self.cube[face][0][1] = old[1][0]
        self.cube[face][0][2] = old[0][0]
        self.cube[face][1][0] = old[2][1]
        self.cube[face][1][1] = old[1][1]
        self.cube[face][1][2] = old[0][1]
        self.cube[face][2][0] = old[2][2]
        self.cube[face][2][1] = old[1][2]
        self.cube[face][2][2] = old[0][2]
        
    def _rotate_face_prime(self, face):
        """Rotate a face 90 degrees counter-clockwise"""
        old = [row[:] for row in self.cube[face]]
        self.cube[face][0][0] = old[0][2]
        self.cube[face][0][1] = old[1][2]
        self.cube[face][0][2] = old[2][2]
        self.cube[face][1][0] = old[0][1]
        self.cube[face][1][1] = old[1][1]
        self.cube[face][1][2] = old[2][1]
        self.cube[face][2][0] = old[0][0]
        self.cube[face][2][1] = old[1][0]
        self.cube[face][2][2] = old[2][0]

    def _calculate_heuristic(self):
        """Calculate advanced heuristic combining multiple factors"""
        corner_score = self._corner_heuristic()
        edge_score = self._edge_heuristic() 
        cross_score = self._cross_heuristic()
        center_score = self._center_heuristic()
        
        self.phi = (cross_score * 3.0 + corner_score * 2.0 + edge_score * 1.5 + center_score * 0.5)
    
    def _corner_heuristic(self):
        """Score based on corner piece positions and orientations"""
        score = 0
        corner_positions = [
            [('U',0,0), ('L',0,0), ('B',0,2)], [('U',0,2), ('B',0,0), ('R',0,2)],
            [('U',2,0), ('F',0,0), ('L',0,2)], [('U',2,2), ('R',0,0), ('F',0,2)],
            [('D',0,0), ('L',2,2), ('F',2,0)], [('D',0,2), ('F',2,2), ('R',2,0)],
            [('D',2,0), ('B',2,0), ('L',2,0)], [('D',2,2), ('R',2,2), ('B',2,2)]
        ]
        
        target_corners = [
            [0,4,3], [0,3,1], [0,2,4], [0,1,2],
            [5,4,2], [5,2,1], [5,3,4], [5,1,3]
        ]
        
        for i, corner_pos in enumerate(corner_positions):
            current = [self.cube[face][row][col] for face, row, col in corner_pos]
            target = target_corners[i]
            
            if set(current) != set(target):
                score += 3
            else:
                if current != target:
                    score += 1
        
        return score
    
    def _edge_heuristic(self):
        """Score based on edge piece positions and orientations"""
        score = 0
        edge_positions = [
            [('U',0,1), ('B',0,1)], [('U',1,2), ('R',0,1)], 
            [('U',2,1), ('F',0,1)], [('U',1,0), ('L',0,1)],
            [('F',1,0), ('L',1,2)], [('F',1,2), ('R',1,0)],
            [('B',1,0), ('R',1,2)], [('B',1,2), ('L',1,0)],
            [('D',0,1), ('F',2,1)], [('D',1,2), ('R',2,1)],
            [('D',2,1), ('B',2,1)], [('D',1,0), ('L',2,1)]
        ]
        
        target_edges = [
            [0,3], [0,1], [0,2], [0,4],
            [2,4], [2,1], [3,1], [3,4],
            [5,2], [5,1], [5,3], [5,4]
        ]
        
        for i, edge_pos in enumerate(edge_positions):
            current = [self.cube[face][row][col] for face, row, col in edge_pos]
            target = target_edges[i]
            
            if set(current) != set(target):
                score += 2
            else:
                if current != target:
                    score += 1
        
        return score
    
    def _cross_heuristic(self):
        """Score for having cross patterns on top and bottom faces"""
        score = 0
        
        top_cross = [self.cube['U'][0][1], self.cube['U'][1][0], 
                    self.cube['U'][1][2], self.cube['U'][2][1]]
        if not all(piece == 0 for piece in top_cross):
            score += 8
            
        bottom_cross = [self.cube['D'][0][1], self.cube['D'][1][0],
                       self.cube['D'][1][2], self.cube['D'][2][1]]
        if not all(piece == 5 for piece in bottom_cross):
            score += 4
            
        return score
    
    def _center_heuristic(self):
        """Score based on center pieces"""
        score = 0
        centers = {'U': 0, 'R': 1, 'F': 2, 'D': 5, 'L': 4, 'B': 3}
        
        for face, target_color in centers.items():
            if self.cube[face][1][1] != target_color:
                score += 10
                
        return score

    def make_move(self, move):
        """Make a move and return move info and heuristic change"""
        if move not in self.moves:
            return None, 0
            
        old_phi = self.phi
        old_state = copy.deepcopy(self.cube)
        
        self._execute_move(move)
        self._calculate_heuristic()
        
        move_info = (move, old_state)
        return move_info, self.phi - old_phi

    def unmake_move(self, move_info):
        """Undo a move"""
        move, old_state = move_info
        self.cube = old_state
        self._calculate_heuristic()

    def get_valid_moves(self):
        """Return all possible moves"""
        return self.moves[:]

    def display(self):
        """Display the cube state in unfolded format"""
        color_map = {0: 'W', 1: 'R', 2: 'B', 3: 'O', 4: 'G', 5: 'Y'}
        
        result = ""
        # Top face (U)
        for i in range(3):
            result += "      " + " ".join(color_map[self.cube['U'][i][j]] for j in range(3)) + "\n"
        result += "\n"
        
        # Middle row: L F R B
        for i in range(3):
            line = ""
            for face in ['L', 'F', 'R', 'B']:
                line += " ".join(color_map[self.cube[face][i][j]] for j in range(3)) + " "
            result += line + "\n"
        result += "\n"
        
        # Bottom face (D)
        for i in range(3):
            result += "      " + " ".join(color_map[self.cube['D'][i][j]] for j in range(3)) + "\n"
        
        return result