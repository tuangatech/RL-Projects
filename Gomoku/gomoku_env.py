import gym
import numpy as np
import re
from gym import spaces
from game_config import Config

# Precompile patterns once for faster matching
COMPILED_PATTERNS = {
    pattern: re.compile(pattern.replace('-', r'\-').replace('o', r'o').replace('X', r'X'))
    for pattern in Config.PATTERN_SCORES
}

class GomokuEnv(gym.Env):

    def __init__(self, board_size=6, win_length=4):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.win_length = win_length
        self.action_space = spaces.Discrete(board_size * board_size) # all possible moves

        # 4 channels: [player's stones, opponent's stones, player's win threat, opponent's win threat]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, board_size, board_size), dtype=np.uint8
        )

        self.reset() # Initialize the environment
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int) # clear the board
        self.current_player = 1  # 1 for player 1, -1 for player 2 (opponent)
        self.done = False
        self.valid_actions = set(range(self.board_size * self.board_size)) # reset valid actions
        return self._get_obs()   # return the initial observation

    # Executes one step in the environment based on the given action
    def step(self, action):
        # Converts the action (a flattened index) into row and column indices
        row, col = divmod(action, self.board_size)

        # If the action is invalid (game is over or cell is already occupied)
        if self.done or self.board[row, col] != 0:
            # self.done = True  # *** End the game if invalid move
            # self.done can be True or False, but the action is invalid
            return self._get_obs(), Config.REWARD_INVALID_MOVE, self.done, {"invalid": True}  # invalid move, reward = -1.0
        
        opponent = -self.current_player

        # Before move: calculate threats and soon-win threats
        player_threat_before = self._calculate_global_threat(self.current_player)
        opponent_threat_before = self._calculate_global_threat(opponent)
        # player_win_threats_before = np.sum(self._get_win_threat_map(self.current_player))
        # opponent_win_threats_before = np.sum(self._get_win_threat_map(opponent))

        # Place the current player's stone on the board
        self.board[row, col] = self.current_player
        # Remove the chosen action from valid actions set
        if action in self.valid_actions: # Defensive check
             self.valid_actions.remove(action)
        
        # --- Check for Terminal States AFTER placing the stone ---

        # Win check
        if self._check_win(row, col, self.current_player):
            self.done = True
            return self._get_obs(), Config.REWARD_WIN, self.done, {"winner": self.current_player}
            
        # Draw check
        if len(self.valid_actions) == 0:  # Board is full
            self.done = True
            return self._get_obs(), Config.REWARD_DRAW, self.done, {"draw": True}
        
        # --- If the game is not terminal (neither win nor draw from this move) ---
        # After move: update threats       
        player_threat_after = self._calculate_global_threat(self.current_player)
        opponent_threat_after = self._calculate_global_threat(opponent)
        # Count where the player who just moved can win next turn
        # player_win_threats_after = np.sum(self._get_win_threat_map(self.current_player))
        # Count where the *opponent* (next player) can win next turn
        # opponent_win_threats_after = np.sum(self._get_win_threat_map(opponent))
        # How many opponent winning chances we blocked by that move
        # blocked_threats = opponent_win_threats_before - opponent_win_threats_after
        # How many new winning chances we created for the player who just moved
        # added_threats = player_win_threats_after - player_win_threats_before

        # Reward Calculation Components (based on the player who just moved)

        # Reward for increasing own threats
        player_threat_delta_reward = player_threat_after - player_threat_before
        # Reward for reducing opponent threats (Strategic Defense)
        opponent_threat_delta_reward = opponent_threat_before - opponent_threat_after

        # Reward for creating forks (using the new fork check)
        fork_reward = Config.REWARD_FORK if self._check_fork(row, col, self.current_player) else 0 

        # Punish for a passive move (no change in threats)
        # pushes the agent to take meaningful actions rather than random placements
        passive_reward = Config.REWARD_PASSIVE if player_threat_delta_reward == 0 and opponent_threat_delta_reward == 0 else 0

        # Reward if after this move, there are new cells where current player can win next move - 
        # already be rewarded by LIVE3 and FORK patterns (immediate win)
        # soon_win_reward = Config.REWARD_SOON_WIN if added_threats > 0 else 0  #  * win_threats_after
        # block_soon_win_reward = Config.REWARD_BLOCK_SOON_WIN if blocked_threats > 0 else 0  #  * blocked_threats

        reward = (player_threat_delta_reward + opponent_threat_delta_reward + fork_reward + passive_reward)
                #   soon_win_reward + block_soon_win_reward
        # print(f"Reward: {reward}, \n" \
        #       f"Player Threat Delta: {player_threat_delta_reward}, Opponent Threat Delta: {opponent_threat_delta_reward}, \n" \
        #       f"Fork: {fork_reward}, Block Soon Win: {block_soon_win_reward}, Passive: {passive_reward} \n")

        # Self-play training: switch turns to the other player
        # The reward is always from the perspective of the current player
        self.current_player *= -1  

        return self._get_obs(), reward, self.done, {} # return observation (current state)

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
        """
        Returns the current observation including raw board state and threat maps.
        Shape: (num_channels, board_size, board_size)
        Channels: [Current Player Stones, Opponent Stones, Current Player Win Threat, Opponent Win Threat]
        """
        # Create binary layers for stones
        my_stone = (self.board == self.current_player).astype(np.uint8)
        opp_stone = (self.board == -self.current_player).astype(np.uint8)

        # Compute threat maps from each player's perspective
        # my_threat = self._get_win_threat_map(self.current_player)
        # opp_threat = self._get_win_threat_map(-self.current_player)

        return np.stack([my_stone, opp_stone], axis=0)  # , my_threat, opp_threat

    # Check if the current player has won by placing a stone at (row, col)
    def _check_win(self, row, col, player):
        # Check horizontal, vertical, diagonal / and \
        win_length = self.win_length
        board = self.board
        board_size = self.board_size

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # (dr, dc)

        for dr, dc in directions:
            count = 0
            # Check in positive direction
            for i in range(-win_length + 1, win_length): # Check within a window around (r, c)
                nr, nc = row + i * dr, col + i * dc
                if 0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == player:
                    count += 1
                    if count >= win_length:
                        return True
                else:
                    count = 0 # Reset count if sequence is broken or out of bounds
        return False
    
    # Helper to extract a sequence of board values along a direction, handling boundaries
    def _get_sequence(self, r, c, dr, dc, length, player):
        """
        Extracts a sequence of board states along a direction starting from (r, c).
        Boundaries are treated as opponent's stones.
        Returns a string: 'X' for player, 'o' for opponent, '-' for empty, 'o' for out of bounds.
        """
        sequence = ""
        board_size = self.board_size
        opponent = -player

        for i in range(length):
            nr = r + i * dr
            nc = c + i * dc

            if 0 <= nr < board_size and 0 <= nc < board_size:
                if self.board[nr, nc] == player:
                    sequence += "X"
                elif self.board[nr, nc] == opponent:
                    sequence += "o"
                else: # self.board[nr, nc] == 0
                    sequence += "-"
            else:
                # Out of bounds is like opponent's stone - it blocks the line
                sequence += "o"
        return sequence
    
    # Heuristic evaluation of the board state after the current player takes a move
    # give higher scores to longer lines 
    # give higher score to lines with two open ends (double attack) compared to lines with only one open end
    def _calculate_global_threat(self, player):
        scored_patterns = set()
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        board_size = self.board_size
        max_pattern_length = max(len(p) for p in Config.PATTERN_SCORES)
        min_pattern_length = min(len(p) for p in Config.PATTERN_SCORES)        

        for r in range(board_size):
            for c in range(board_size):
                for dr, dc in directions:
                    sequence = self._get_sequence(r, c, dr, dc, max_pattern_length, player)

                    # Sequence must be at least long enough to contain any pattern
                    if len(sequence) < min_pattern_length:
                        continue
                    
                    # Find all occurrences of each pattern using regex within the sequence
                    for pattern, regex in COMPILED_PATTERNS.items():
                        for match in regex.finditer(sequence):
                            i = match.start()
                            start_r = r + i * dr
                            start_c = c + i * dc
                            # Create a unique key for this specific pattern instance found on the board.
                            key = (pattern, start_r, start_c, dr, dc)
                            # Add the key to the set. Set handles uniqueness automatically.
                            scored_patterns.add(key)
        # if scored_patterns:
        #     for pattern, *_ in scored_patterns:
        #         print(f"Pattern: {pattern}, score: {Config.PATTERN_SCORES[pattern]}")
        return sum(Config.PATTERN_SCORES[pattern] for pattern, *_ in scored_patterns)

    
    # Check if the move at (row, col) creates a fork (two+ separate significant threats).
    # In Gomoku game of [6,6,4], a fork is defined by creating two or more Live 3 or Semi-Open 3 configurations.
    def _check_fork(self, row, col, player):
        """
        Check if the move at (row, col) creates a fork (at least two distinct
        Live 3s or Semi-Open 3s involving the placed stone).
        """
        fork_threat_count = 0
        board_size = self.board_size
        max_pattern_length = max(len(p) for p in Config.PATTERN_SCORES)

        # Define the patterns that constitute a significant threat for a fork
        fork_patterns = ["-XXX-", "-XXX", "XXX-", "XX-X", "X-XX", "XX-XX"]

        # Directions to check
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # We already placed the stone in the step function before calling this,
        # so we are checking the state *after* the move.
        # Check each direction for newly formed threats involving the cell (row, col)
        for dr, dc in directions:
            found_threat_in_this_direction = False
            for start_offset in range(-(max_pattern_length - 1), 1):
                start_r = row + start_offset * dr
                start_c = col + start_offset * dc
                # We need to check for patterns in segments of length `check_length` starting at `(r - i*dr, c - i*dc)`
                # where `i` ranges from 0 to `check_length - 1`.

                i = -start_offset # The index of (row, col) in the sequence starting at (start_r, start_c)

                # Get the sequence starting from (start_r, start_c)
                sequence = self._get_sequence(start_r, start_c, dr, dc, max_pattern_length, player)

                # Check if any of the fork patterns are in this sequence
                for pattern in fork_patterns:
                    if pattern in sequence:
                        found_threat_in_this_direction = True
                    #   print(f">> pattern: {pattern} found in sequence: {sequence} at ({start_r}, {start_c})")
                        break # Found a threat pattern in this direction, move to the next direction

            if found_threat_in_this_direction:
                fork_threat_count += 1
        # A fork is created if the move resulted in at least two distinct lines
        # containing significant threats (Live 3 or Semi-Open 4).
        return fork_threat_count >= 2
    
    # def _check_win_threat(self, r, c, player):
    #     """
    #     Checks if placing a stone at (r, c) results in a win for the player.
    #     Assumes (r, c) is currently empty.
    #     """
    #     # Temporarily place the stone
    #     self.board[r, c] = player
    #     win = self._check_win(r, c, player)
    #     # Remove the stone
    #     self.board[r, c] = 0
    #     return win
    
    # # see if placing a stone there would result in an immediate win,
    # # providing the agent with information about both sides' immediate threats
    # # Feeding threat maps into the observation space could help guide early 
    # # learning by acting as a "cheat code" for detecting critical positions.
    # def _get_win_threat_map(self, player):
    #     """
    #     Creates a binary map indicating where the player can win immediately.
    #     """
    #     threat_map = np.zeros((self.board_size, self.board_size), dtype=np.uint8)

    #     # Iterate through all empty cells
    #     empty_cells = np.argwhere(self.board == 0)

    #     for r, c in empty_cells:
    #         if self._check_win_threat(r, c, player):
    #             threat_map[r, c] = 1

    #     return threat_map
