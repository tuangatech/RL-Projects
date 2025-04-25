import gym
import numpy as np
from gym import spaces
from game_config import Config

class GomokuEnv(gym.Env):

    def __init__(self, board_size=6, win_length=4):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.win_length = win_length
        self.action_space = spaces.Discrete(board_size * board_size) # all possible moves

        # 2 channels: [player's stones, opponent's stones]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, board_size, board_size), dtype=np.uint8
        )

        self._line_cache = {}  # Cache for line results
        self.last_action = None  # Last action taken
        self.valid_actions = set(range(self.board_size * self.board_size))

        self.reset() # Initialize the environment
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int) # clear the board
        self.current_player = 1  # 1 for player 1, -1 for player 2 (opponent)
        self.done = False
        self.last_action = None
        self._line_cache.clear()  # Clear the cache for new game
        self.valid_actions = set(range(self.board_size * self.board_size)) # reset valid actions
        return self._get_obs()   # return the initial observation

    # Executes one step in the environment based on the given action
    def step(self, action):
        # Converts the action (a flattened index) into row and column indices
        row, col = divmod(action, self.board_size)
        # Store the last action for cache clearing
        self.last_action = action   

        # If the action is invalid (game is over or cell is already occupied)
        if self.done or self.board[row, col] != 0:
            return self._get_obs(), -1.0, True, {"invalid": True}  # invalid move, reward = -1.0
        
        # Calculate opponent's threat before the move
        opponent = -self.current_player
        opponent_threat_before = self._check_threat(opponent)

        # Place the current player's stone on the board
        self.board[row, col] = self.current_player
        # Remove the chosen action from valid actions set
        self.valid_actions.remove(action)
        # Clear cache only for affected cells in the radius of 3x3 around the last move (last_action)
        self._clear_cache_around()

        reward = 0
        done = False
        info = {}

        # If the move results in a win, set the reward and mark the game as done
        if self._check_win(row, col, self.current_player):
            reward = Config.REWARD_WIN
            done = True
        # If the board is full and no one has won, mark the game as done
        elif np.all(self.board != 0):
            reward = Config.REWARD_DRAW
            done = True
        else:
            # Evaluate the board state after the move
            reward = self._evaluate_board(self.current_player)
            # Encourage moves near existing stones
            reward += self._proximity_reward()  
            # Reward for Strategic Defense
            opponent_threat_after = self._check_threat(opponent)
            reward += opponent_threat_before - opponent_threat_after

            # Reward for Forks
            if self._check_fork(row, col, self.current_player):
                reward += Config.REWARD_FORK  # Fixed reward for creating forks

            self.current_player *= -1  # switch turns

        self.done = done
        return self._get_obs(), reward, done, info # return observation (current state)

    # Render the current state of the board in a human-readable format
    # This function is useful for debugging and visualization
    def render(self, mode="human"):
        symbols = {0: ".", 1: "X", -1: "O"}
        for row in self.board:
            print(" ".join(symbols[val] for val in row))
        print()

    # Get all valid actions (empty cells) for the current player
    # This function is useful for agents to know where they can place their stones
    def get_valid_actions(self):
        return list(self.valid_actions)

    # Generates the observation (state) for the current player
    def _get_obs(self):
        # Two planes: one for current player's stones, one for opponent's
        if self.current_player == 1:
            current = (self.board == 1).astype(np.uint8)
            opponent = (self.board == -1).astype(np.uint8)
        else:
            current = (self.board == -1).astype(np.uint8)
            opponent = (self.board == 1).astype(np.uint8)
        return np.stack([current, opponent])  # stack the two planes (2, board_size, board_size)

    # Check if the current player has won by placing a stone at (row, col)
    def _check_win(self, row, col, player):
        # checks all four possible winning directions (horizontal, vertical, diagonal)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in [-1, 1]:
                r, c = row, col
                # counts consecutive stones of the same player
                while True:
                    r += sign * dr
                    c += sign * dc
                    if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                        count += 1
                    else:
                        break
            if count >= self.win_length:
                return True
        return False

    # Heuristic evaluation of the board state after the current player takes a move
    # give higher scores to longer lines 
    # give higher score to lines with two open ends (double attack) compared to lines with only one open end
    def _evaluate_board(self, player):    
        # difference between the player's score and the opponent's score
        return self._check_threat(player) - self._check_threat(-player)
    
    # Simple reward shaping: +0.01 if 2 or 3 in a row with open ends, -0.01 if opponent has it
    def _check_threat(self, p):
        # Evaluate only the cells near the last move
        row, col = divmod(self.last_action, self.board_size)
        radius = Config.IMPACT_RADIUS  # Consider a 3x3 region around the last move
        score = 0.0
        for r in range(max(0, row - radius), min(self.board_size, row + radius + 1)):
            for c in range(max(0, col - radius), min(self.board_size, col + radius + 1)):
                # if the line has 2 stones in a row and 2 open ends
                if self._check_line(p, length=2, open_ends=2, r=r, c=c):
                    score += Config.REWARD_LENGTH2
                if self._check_line(p, length=3, open_ends=1, r=r, c=c):
                    score += Config.REWARD_LENGTH3
                if self._check_line(p, length=3, open_ends=2, r=r, c=c):
                    score += Config.REWARD_LENGTH4
                # if self._check_line(p, length=4, open_ends=1, r=r, c=c):
                #     score += Config.SCORE_LENGTH4
        return score
        
    # Check if there are 'length' stones in a row and at least one open end
    # This function is used to evaluate the board state
    def _check_line(self, player, length, open_ends, r, c):
        # Generate a unique key for the line check
        cache_key = (player, length, open_ends, r, c)
        if cache_key in self._line_cache:
            return self._line_cache[cache_key]

        result = False

        # Check in all four directions
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            open_end_pos = False
            open_end_neg = False

            # Check in the positive direction
            for k in range(1, length):
                nr = r + k * dr
                nc = c + k * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr][nc] == player:
                        count += 1
                    elif self.board[nr][nc] == 0:
                        open_end_pos = True
                    else:
                        break
                else:
                    open_end_pos = True
                    break

            # Check in the negative direction
            for k in range(1, length):
                nr = r - k * dr
                nc = c - k * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr][nc] == player:
                        count += 1
                    elif self.board[nr][nc] == 0:
                        open_end_neg = True
                    else:
                        break
                else:
                    open_end_neg = True
                    break

            # Check if the line satisfies the required conditions
            if count == length:
                if open_ends == 1 and (open_end_pos or open_end_neg):
                    result = True
                    break
                if open_ends == 2 and open_end_pos and open_end_neg:
                    result = True
                    break
        # Cache the result for future reference
        self._line_cache[cache_key] = result

        return result

    # Encourage moves near existing stones (player's or opponent's)
    # This function is used to promote more strategic placements of stones
    def _proximity_reward(self):
        row, col = divmod(self.last_action, self.board_size)
        start_r, end_r = max(0, row - 1), min(self.board_size, row + 2)
        start_c, end_c = max(0, col - 1), min(self.board_size, col + 2)
        neighborhood = self.board[start_r:end_r, start_c:end_c]
        if np.any(neighborhood != 0):
            return Config.REWARD_PROXIMITY # Small reward for proximity to existing stones
        return 0.0

    def _check_fork(self, row, col, player):
        """
        Check if the move at (row, col) creates a fork (two+ separate lines of 3 stones).
        Lines can be semi-open (only one open end required).
        """
        directions = [
            (0, 1),   # Horizontal
            (1, 0),    # Vertical
            (1, 1),    # Diagonal down-right
            (1, -1),   # Diagonal down-left
        ]
        
        board = self.board
        board_size = self.board_size
        fork_lines = 0
        
        for dr, dc in directions:
            # Check both directions along each axis
            line_pos = []
            line_neg = []
            
            # Positive direction (e.g., right for horizontal)
            for i in range(1, 4):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < board_size and 0 <= c < board_size:
                    line_pos.append(board[r, c])
                else:
                    line_pos.append(-2)  # Out of bounds (blocked)
            
            # Negative direction (e.g., left for horizontal)
            for i in range(1, 4):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < board_size and 0 <= c < board_size:
                    line_neg.append(board[r, c])
                else:
                    line_neg.append(-2)  # Out of bounds (blocked)
            
            # Check for sequences of 2 stones in either direction
            # Combined with the new stone, this makes 3 in a row
            if (len(line_pos) >= 2 and all(s == player for s in line_pos[:2])):
                fork_lines += 1
            if (len(line_neg) >= 2 and all(s == player for s in line_neg[:2])):
                fork_lines += 1
            
            # Early exit if we already found a fork
            if fork_lines >= 2:
                return True
        
        return False  # A fork is created if there are at least two open-ended lines of length 3
    
    def _clear_cache_around(self):
        """
        Clear cache entries for cells within a radius of 3x3 around the given cell.
        This ensures that affected areas are recalculated while retaining results for unchanged cells.
        """
        row, col = divmod(self.last_action, self.board_size)  # Get the last move's position
        radius = Config.IMPACT_RADIUS  # Radius around the last move (row, col) to clear
        for r in range(max(0, row - radius), min(self.board_size, row + radius + 1)):
            for c in range(max(0, col - radius), min(self.board_size, col + radius + 1)):
                for player in [-1, 1]:
                    for length in [2, 3]:  # Lengths relevant to threat evaluation, 4 is not used in this context of 6x6 board
                        for open_ends in [1, 2]:  # Open-end conditions
                            cache_key = (player, length, open_ends, r, c)
                            if cache_key in self._line_cache:
                                del self._line_cache[cache_key]