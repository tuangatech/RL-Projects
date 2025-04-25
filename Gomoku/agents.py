import numpy as np
import random


class RandomAgent:
    def __init__(self, board_size):
        self.board_size = board_size

    def act(self, obs):
        board = (obs[0] + obs[1]).reshape(-1)
        valid_actions = [i for i, val in enumerate(board) if val == 0]
        return random.choice(valid_actions)


class GreedyAgent:
    def __init__(self, board_size):
        self.board_size = board_size
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # horizontal, vertical, diagonal, anti-diagonal
        
    def act(self, obs):
        my_stones = obs[0]
        opponent_stones = obs[1]
        combined_board = my_stones + opponent_stones
        valid_moves = np.where(combined_board.reshape(-1) == 0)[0]
        
        if len(valid_moves) == 0:
            return None
            
        # First move: play center
        if np.sum(combined_board) == 0:
            center = self.board_size // 2
            return center * self.board_size + center
        
        # Calculate scores for each valid move
        scores = np.zeros(self.board_size * self.board_size)
        
        for move in valid_moves:
            x, y = divmod(move, self.board_size)
            
            # Check all directions around this position
            for dx, dy in self.directions:
                # Count my stones in both directions
                my_count = 0
                for i in range(1, 5):
                    nx, ny = x + dx * i, y + dy * i
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if my_stones[nx, ny] == 1:
                            my_count += 1
                        elif combined_board[nx, ny] == 1:  # opponent stone
                            break
                    else:
                        break
                        
                for i in range(1, 5):
                    nx, ny = x - dx * i, y - dy * i
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if my_stones[nx, ny] == 1:
                            my_count += 1
                        elif combined_board[nx, ny] == 1:  # opponent stone
                            break
                    else:
                        break
                
                # Add score based on my stones
                scores[move] += my_count * 2
                
                # Count opponent stones in both directions (for blocking)
                opp_count = 0
                for i in range(1, 5):
                    nx, ny = x + dx * i, y + dy * i
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if opponent_stones[nx, ny] == 1:
                            opp_count += 1
                        elif combined_board[nx, ny] == 1:  # my stone
                            break
                    else:
                        break
                        
                for i in range(1, 5):
                    nx, ny = x - dx * i, y - dy * i
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if opponent_stones[nx, ny] == 1:
                            opp_count += 1
                        elif combined_board[nx, ny] == 1:  # my stone
                            break
                    else:
                        break
                
                # Add score based on opponent stones (for blocking)
                # Score blocking more aggressively if opponent has 3 or 4 in a row
                if opp_count >= 3:
                    scores[move] += opp_count * 3
                else:
                    scores[move] += opp_count
            
            # Small bonus for center proximity as tiebreaker
            center = self.board_size // 2
            dist = abs(x - center) + abs(y - center)
            scores[move] += 0.1 / (1 + dist)
        
        # Choose move with highest score
        best_move = np.argmax(scores)
        return best_move