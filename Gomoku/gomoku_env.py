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

        self.last_action = None  # Last action taken
        self.valid_actions = set(range(self.board_size * self.board_size))

        self.reset() # Initialize the environment
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int) # clear the board
        self.current_player = 1  # 1 for player 1, -1 for player 2 (opponent)
        self.done = False
        self.last_action = None
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
            return self._get_obs(), Config.REWARD_INVALID_MOVE, True, {"invalid": True}  # invalid move, reward = -1.0
        
        # Calculate opponent's threat before the move
        opponent = -self.current_player
        opponent_threat_before = self._calculate_global_threat(opponent)
        player_threat_before = self._calculate_global_threat(self.current_player) # Also calculate player's threat before

        # Place the current player's stone on the board
        self.board[row, col] = self.current_player
        # Remove the chosen action from valid actions set
        self.valid_actions.remove(action)
        
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
            player_threat_after = self._calculate_global_threat(self.current_player)
            opponent_threat_after = self._calculate_global_threat(opponent)

            # Reward for increasing own threats
            reward += player_threat_after - player_threat_before
            # print(f">> Player Threat Before: {player_threat_before}, After: {player_threat_after} > reward: {reward}")
            # Reward for reducing opponent threats (Strategic Defense)
            reward += opponent_threat_before - opponent_threat_after
            # print(f">> Opponent Threat Before: {opponent_threat_before}, After: {opponent_threat_after} > reward: {reward}")

            # Reward for creating forks (using the new fork check)
            if self._check_fork(row, col, self.current_player):
                reward += Config.REWARD_FORK
                # print(f">> Fork Detected! + {Config.REWARD_FORK} > reward: {reward}")

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
        """
        Returns the current observation including raw board state and threat maps.
        Shape: (num_channels, board_size, board_size)
        Channels: [Current Player Stones, Opponent Stones, Current Player Win Threat, Opponent Win Threat]
        """
        obs_raw = np.zeros((2, self.board_size, self.board_size), dtype=np.uint8)
        obs_raw[0, self.board == 1] = 1 # Player 1's stones
        obs_raw[1, self.board == -1] = 1 # Player -1's stones

        current_player_threat_map = self._get_win_threat_map(self.current_player)
        opponent_threat_map = self._get_win_threat_map(-self.current_player)

        # Stack the channels. Ensure current player's view is always first.
        if self.current_player == 1:
            obs = np.stack([
                obs_raw[0],               # P1 Stones (Current)
                obs_raw[1],               # P-1 Stones (Opponent)
                current_player_threat_map, # P1 Win Threat (Current)
                opponent_threat_map       # P-1 Win Threat (Opponent)
            ], axis=0)
        else: # self.current_player == -1
            obs = np.stack([
                obs_raw[1],               # P-1 Stones (Current)
                obs_raw[0],               # P1 Stones (Opponent)
                current_player_threat_map, # P-1 Win Threat (Current)
                opponent_threat_map       # P1 Win Threat (Opponent) - Note: Threat maps are calculated for the absolute player
            ], axis=0)

        # Ensure the threat maps in the observation are always from the perspective
        # of the channels they are paired with (channel 0/1 is current/opponent).
        # Let's adjust the stacking slightly for clarity:
        p1_stones = (self.board == 1).astype(np.uint8)
        p_minus1_stones = (self.board == -1).astype(np.uint8)
        p1_threat = self._get_win_threat_map(1)
        p_minus1_threat = self._get_win_threat_map(-1)

        if self.current_player == 1:
            obs = np.stack([
                p1_stones,         # Channel 0: Current Player Stones
                p_minus1_stones,   # Channel 1: Opponent Stones
                p1_threat,         # Channel 2: Current Player Win Threat
                p_minus1_threat    # Channel 3: Opponent Win Threat
            ], axis=0)
        else: # self.current_player == -1
            obs = np.stack([
                p_minus1_stones,   # Channel 0: Current Player Stones
                p1_stones,         # Channel 1: Opponent Stones
                p_minus1_threat,   # Channel 2: Current Player Win Threat
                p1_threat          # Channel 3: Opponent Win Threat
            ], axis=0)


        return obs

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

    # Heuristic evaluation of the board state after the current player takes a move
    # give higher scores to longer lines 
    # give higher score to lines with two open ends (double attack) compared to lines with only one open end
    def _evaluate_board(self, player):    
        # difference between the player's score and the opponent's score
        return self._calculate_global_threat(player) - self._calculate_global_threat(-player)
    
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
    
        # Calculates the global threat score for a player by checking patterns across the board
    def _calculate_global_threat(self, player):
        """
        Evaluates the board state for a given player by finding unique instances of patterns
        (Live 2, Semi-Open 3, Live 3, Semi-Open 4) globally.
        Uses a set to ensure each pattern instance on the board is counted only once.
        A pattern instance is identified by its type, its starting board coordinates, and its direction.
        """
        # Use a set to store unique pattern instances found.
        # A pattern instance is uniquely identified by a tuple:
        # (pattern_string, starting_row, starting_col, direction_dr, direction_dc).
        # We use the board coordinates corresponding to the START of the matched pattern substring.
        scored_patterns = set()

        board_size = self.board_size
        # CHECK_SEQUENCE_LENGTH needs to be large enough to contain the longest pattern + its open ends.
        # "-XXXXo" is length 6. "-XXX-" is length 5. Let's use 6.
        check_length = max(len(p) for p in Config.PATTERN_SCORES.keys()) # Ensure CHECK_SEQUENCE_LENGTH is sufficient

        # Directions to check (only need to check in one direction per axis to cover all lines)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # Iterate through all possible starting cells (r, c) for a sequence of length `check_length`.
        # Any sequence of length `check_length` that starts within the board bounds
        # (considering _get_sequence handles padding for going slightly out of bounds)
        # will be checked. Patterns might occur at any index within such a sequence.
        for r in range(board_size):
            for c in range(board_size):
                # For each starting cell (r, c), check in the 4 canonical directions
                for dr, dc in directions:
                    # Get the sequence starting from (r, c) in direction (dr, dc) with length check_length.
                    # _get_sequence handles out-of-bounds padding.
                    sequence = self._get_sequence(r, c, dr, dc, check_length, player)

                    # Find all occurrences of each pattern within this generated sequence
                    for pattern, pattern_score in Config.PATTERN_SCORES.items():
                        pattern_len = len(pattern)
                        # Use find() in a loop to get all occurrences of the pattern substring within the sequence.
                        # Start searching from index 0 of the sequence.
                        match_index = sequence.find(pattern)

                        while match_index != -1:
                            # Calculate the board coordinates of the START of the matched pattern instance.
                            # The match_index is the index within the sequence string where the pattern begins.
                            # The cell (r, c) from the outer loop corresponds to index 0 in the sequence.
                            # So, the board cell corresponding to match_index in the sequence is located at:
                            # (r + match_index * dr, c + match_index * dc).
                            pattern_start_r = r + match_index * dr
                            pattern_start_c = c + match_index * dc

                            # Create a unique key for this specific pattern instance found on the board.
                            # The key includes the pattern string itself, its exact starting board coordinates,
                            # and the direction it lies in.
                            # Using the canonical direction (dr, dc) ensures uniqueness across different sequences
                            # that might contain the same pattern instance.
                            pattern_instance_key = (pattern, pattern_start_r, pattern_start_c, dr, dc)

                            # Add the key to the set. If this exact instance (pattern at this location in this direction)
                            # has already been found and added from a different sequence starting point, the set
                            # automatically ignores the duplicate addition.
                            scored_patterns.add(pattern_instance_key)

                            # Find the next occurrence of the pattern in the current sequence,
                            # starting the search one character after the beginning of the last match.
                            # This prevents infinite loops if the pattern is empty or found at index 0 repeatedly
                            # (though our patterns are not empty). It ensures we look for subsequent matches.
                            match_index = sequence.find(pattern, match_index + 1)

        # Calculate the total score by summing the scores of all unique pattern instances found.
        total_score = 0.0
        for pattern, start_r, start_c, dr, dc in scored_patterns:
             # Retrieve the score for the pattern type from the Config.
             # Use .get() with a default of 0.0 for safety, although the pattern should always be a key in PATTERN_SCORES.
             total_score += Config.PATTERN_SCORES.get(pattern, 0.0)

        return total_score
    
    # Check if the move at (row, col) creates a fork (two+ separate significant threats).
    # In Gomoku game of [6,6,4], a fork is defined by creating two or more Live 3 or Semi-Open 3 configurations.
    def _check_fork(self, row, col, player):
        """
        Check if the move at (row, col) creates a fork (at least two distinct
        Live 3s or Semi-Open 3s involving the placed stone).
        """
        fork_threat_count = 0
        board_size = self.board_size
        check_length = Config.CHECK_SEQUENCE_LENGTH

        # Define the patterns that constitute a significant threat for a fork
        fork_patterns = ["-XXX-", "-XXXo", "oXXX-"]

        # Directions to check
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # Temporarily place the stone to evaluate the resulting state
        # We already placed the stone in the step function before calling this,
        # so we are checking the state *after* the move.

        # Check each direction for newly formed threats involving the cell (row, col)
        for dr, dc in directions:
             # Extract sequence passing through (row, col)
             # Need to check a sequence long enough to contain a pattern centered or near (row, col)
             # Let's extract a sequence of length 7 (3 back, the stone, 3 forward) if possible.
             # A simpler approach is to check segments of CHECK_SEQUENCE_LENGTH that *include* (row, col)

             # Iterate through potential start points such that the segment includes (row, col)
             # A segment starting at (row - i*dr, col - i*dc) of length L includes (row, col) if 0 <= i < L.
             # We need to check segments of length `check_length` that contain (row, col).
             # The potential start indices relative to (row, col) are from -(check_length - 1) to 0.

             found_threat_in_this_direction = False
             for start_offset in range(-(check_length - 1), 1):
                 start_r = row + start_offset * dr
                 start_c = col + start_offset * dc

                 # Check if this starting point is valid (sequence can start here)
                 # It's valid if the whole sequence fits or extends correctly into boundaries
                 # We already handle boundaries in _get_sequence, so just need to check if the start is on board
                 # OR if it's just off board but the sequence comes back on.
                 # A simpler check: if the calculated start_r, start_c, when combined with dr, dc, covers (row, col)
                 # and the sequence length is sufficient.

                 # Let's just iterate through all valid start points that *could* contain (row, col)
                 # A line segment from (sr, sc) with length L in direction (dr, dc) includes (r, c) if
                 # (r, c) = (sr + i*dr, sc + i*dc) for some 0 <= i < L.
                 # This means sr = r - i*dr and sc = c - i*dc.
                 # We need to check for patterns in segments of length `check_length` starting at `(r - i*dr, c - i*dc)`
                 # where `i` ranges from 0 to `check_length - 1`.

                 i = -start_offset # The index of (row, col) in the sequence starting at (start_r, start_c)

                 # Get the sequence starting from (start_r, start_c)
                 sequence = self._get_sequence(start_r, start_c, dr, dc, check_length, player)

                 # Check if any of the fork patterns are in this sequence
                 for pattern in fork_patterns:
                     if pattern in sequence:
                          found_threat_in_this_direction = True
                        #   print(f">> pattern: {pattern} found in sequence: {sequence} at ({start_r}, {start_c})")
                          break # Found a threat pattern in this direction, move to the next direction

             if found_threat_in_this_direction:
                 fork_threat_count += 1
        # print(f">> Fork Threat Count: {fork_threat_count}")
        # A fork is created if the move resulted in at least two distinct lines
        # containing significant threats (Live 3 or Semi-Open 4).
        return fork_threat_count >= 2
    
    def _check_win_threat(self, r, c, player):
        """
        Checks if placing a stone at (r, c) results in a win for the player.
        Assumes (r, c) is currently empty.
        """
        # Temporarily place the stone
        self.board[r, c] = player
        win = self._check_win(r, c, player)
        # Remove the stone
        self.board[r, c] = 0
        return win
    

    def _get_win_threat_map(self, player):
        """
        Creates a binary map indicating where the player can win immediately.
        """
        threat_map = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        board_size = self.board_size

        # Iterate through all empty cells
        empty_cells = np.argwhere(self.board == 0)

        for r, c in empty_cells:
            if self._check_win_threat(r, c, player):
                threat_map[r, c] = 1

        return threat_map

    
    # Simple reward shaping: +0.01 if 2 or 3 in a row with open ends, -0.01 if opponent has it
    # def _check_threat(self, p):
    #     # Evaluate only the cells near the last move
    #     row, col = divmod(self.last_action, self.board_size)
    #     radius = Config.IMPACT_RADIUS  # Consider a 3x3 region around the last move
    #     score = 0.0
    #     for r in range(max(0, row - radius), min(self.board_size, row + radius + 1)):
    #         for c in range(max(0, col - radius), min(self.board_size, col + radius + 1)):
    #             # if the line has 2 stones in a row and 2 open ends
    #             if self._check_line(p, length=2, open_ends=2, r=r, c=c):
    #                 score += Config.REWARD_LENGTH2
    #             if self._check_line(p, length=3, open_ends=1, r=r, c=c):
    #                 score += Config.REWARD_LENGTH3
    #             if self._check_line(p, length=3, open_ends=2, r=r, c=c):
    #                 score += Config.REWARD_LENGTH4
    #             # if self._check_line(p, length=4, open_ends=1, r=r, c=c):
    #             #     score += Config.SCORE_LENGTH4
    #     return score
        
    # Check if there are 'length' stones in a row and at least one open end
    # This function is used to evaluate the board state
    # def _check_line(self, player, length, open_ends, r, c):
    #     # Generate a unique key for the line check
    #     cache_key = (player, length, open_ends, r, c)
    #     if cache_key in self._line_cache:
    #         return self._line_cache[cache_key]

    #     result = False

    #     # Check in all four directions
    #     for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
    #         count = 1
    #         open_end_pos = False
    #         open_end_neg = False

    #         # Check in the positive direction
    #         for k in range(1, length):
    #             nr = r + k * dr
    #             nc = c + k * dc
    #             if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
    #                 if self.board[nr][nc] == player:
    #                     count += 1
    #                 elif self.board[nr][nc] == 0:
    #                     open_end_pos = True
    #                 else:
    #                     break
    #             else:
    #                 open_end_pos = True
    #                 break

    #         # Check in the negative direction
    #         for k in range(1, length):
    #             nr = r - k * dr
    #             nc = c - k * dc
    #             if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
    #                 if self.board[nr][nc] == player:
    #                     count += 1
    #                 elif self.board[nr][nc] == 0:
    #                     open_end_neg = True
    #                 else:
    #                     break
    #             else:
    #                 open_end_neg = True
    #                 break

    #         # Check if the line satisfies the required conditions
    #         if count == length:
    #             if open_ends == 1 and (open_end_pos or open_end_neg):
    #                 result = True
    #                 break
    #             if open_ends == 2 and open_end_pos and open_end_neg:
    #                 result = True
    #                 break
    #     # Cache the result for future reference
    #     self._line_cache[cache_key] = result

    #     return result

    # Encourage moves near existing stones (player's or opponent's)
    # This function is used to promote more strategic placements of stones
    # def _proximity_reward(self):
    #     row, col = divmod(self.last_action, self.board_size)
    #     start_r, end_r = max(0, row - 1), min(self.board_size, row + 2)
    #     start_c, end_c = max(0, col - 1), min(self.board_size, col + 2)
    #     neighborhood = self.board[start_r:end_r, start_c:end_c]
    #     if np.any(neighborhood != 0):
    #         return Config.REWARD_PROXIMITY # Small reward for proximity to existing stones
    #     return 0.0

    # def _check_fork(self, row, col, player):
    #     """
    #     Check if the move at (row, col) creates a fork (two+ separate lines of 3 stones).
    #     Lines can be semi-open (only one open end required).
    #     """
    #     directions = [
    #         (0, 1),   # Horizontal
    #         (1, 0),    # Vertical
    #         (1, 1),    # Diagonal down-right
    #         (1, -1),   # Diagonal down-left
    #     ]
        
    #     board = self.board
    #     board_size = self.board_size
    #     fork_lines = 0
        
    #     for dr, dc in directions:
    #         # Check both directions along each axis
    #         line_pos = []
    #         line_neg = []
            
    #         # Positive direction (e.g., right for horizontal)
    #         for i in range(1, 4):
    #             r, c = row + i * dr, col + i * dc
    #             if 0 <= r < board_size and 0 <= c < board_size:
    #                 line_pos.append(board[r, c])
    #             else:
    #                 line_pos.append(-2)  # Out of bounds (blocked)
            
    #         # Negative direction (e.g., left for horizontal)
    #         for i in range(1, 4):
    #             r, c = row - i * dr, col - i * dc
    #             if 0 <= r < board_size and 0 <= c < board_size:
    #                 line_neg.append(board[r, c])
    #             else:
    #                 line_neg.append(-2)  # Out of bounds (blocked)
            
    #         # Check for sequences of 2 stones in either direction
    #         # Combined with the new stone, this makes 3 in a row
    #         if (len(line_pos) >= 2 and all(s == player for s in line_pos[:2])):
    #             fork_lines += 1
    #         if (len(line_neg) >= 2 and all(s == player for s in line_neg[:2])):
    #             fork_lines += 1
            
    #         # Early exit if we already found a fork
    #         if fork_lines >= 2:
    #             return True
        
    #     return False  # A fork is created if there are at least two open-ended lines of length 3
    
    # def _clear_cache_around(self):
    #     """
    #     Clear cache entries for cells within a radius of 3x3 around the given cell.
    #     This ensures that affected areas are recalculated while retaining results for unchanged cells.
    #     """
    #     row, col = divmod(self.last_action, self.board_size)  # Get the last move's position
    #     radius = Config.IMPACT_RADIUS  # Radius around the last move (row, col) to clear
    #     for r in range(max(0, row - radius), min(self.board_size, row + radius + 1)):
    #         for c in range(max(0, col - radius), min(self.board_size, col + radius + 1)):
    #             for player in [-1, 1]:
    #                 for length in [2, 3]:  # Lengths relevant to threat evaluation, 4 is not used in this context of 6x6 board
    #                     for open_ends in [1, 2]:  # Open-end conditions
    #                         cache_key = (player, length, open_ends, r, c)
    #                         if cache_key in self._line_cache:
    #                             del self._line_cache[cache_key]
    