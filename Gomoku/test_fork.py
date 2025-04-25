from gomoku_env import GomokuEnv
import numpy as np

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
    
    return False

def _check_fork2(self, row, col, player):
    """
    Check if placing a stone at (row, col) creates two open-ended lines of length 3.
    This implementation uses NumPy for vectorized computation.
    """
    # Define all four directions: horizontal, vertical, diagonal, anti-diagonal
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    # Create arrays for positive and negative offsets
    offsets = np.arange(1, self.win_length)  # Offsets for checking neighbors
    drs, dcs = zip(*directions)  # Separate row and column deltas
    drs, dcs = np.array(drs), np.array(dcs)

    # Compute coordinates for positive and negative directions
    pos_rows = row + np.outer(offsets, drs)  # Positive direction rows
    pos_cols = col + np.outer(offsets, dcs)  # Positive direction columns
    neg_rows = row - np.outer(offsets, drs)  # Negative direction rows
    neg_cols = col - np.outer(offsets, dcs)  # Negative direction columns

    # Mask for valid coordinates (within board bounds)
    pos_valid_mask = (pos_rows >= 0) & (pos_rows < self.board_size) & (pos_cols >= 0) & (pos_cols < self.board_size)
    neg_valid_mask = (neg_rows >= 0) & (neg_rows < self.board_size) & (neg_cols >= 0) & (neg_cols < self.board_size)

    # Extract values from the board for valid coordinates
    pos_values = np.where(pos_valid_mask, self.board[pos_rows[pos_valid_mask], pos_cols[pos_valid_mask]], -2)
    neg_values = np.where(neg_valid_mask, self.board[neg_rows[neg_valid_mask], neg_cols[neg_valid_mask]], -2)

    # Count consecutive stones in each direction
    def count_consecutive(values, player):
        counts = np.zeros(len(directions), dtype=int)
        for i in range(len(directions)):
            line = values[:, i]
            mask = (line == player).astype(int)
            counts[i] = np.max(np.cumsum(mask))
        return counts

    pos_counts = count_consecutive(pos_values, player)
    neg_counts = count_consecutive(neg_values, player)

    # Combine counts from both directions
    total_counts = pos_counts + neg_counts

    # Check for open ends
    def has_open_end(values):
        return np.any(values == 0, axis=0)

    pos_open_ends = has_open_end(pos_values)
    neg_open_ends = has_open_end(neg_values)

    # A fork exists if there are at least two open-ended lines of length 3
    open_ended_lines = ((total_counts == 3) & (pos_open_ends | neg_open_ends)).sum()
    return open_ended_lines >= 2

def test_check_fork(env, board, row, col, player, expected_output):
    env.board = board  # Set the board state
    result = env._check_fork(row, col, player)
    print(f"Test at ({row}, {col}) for player {player}:")
    print("Board:")
    print(board)
    print(f"Expected Output: {expected_output}, Actual Output: {result}")
    assert result == expected_output, "Test Failed!"
    print("Test Passed!\n")

# Initialize the environment
env = GomokuEnv(board_size=6, win_length=4)

# Test Case 1: Simple Fork
board = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [-1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])
test_check_fork(env, board, row=2, col=3, player=1, expected_output=True)

# Test Case 2: No Fork
board = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, -1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, -1, 1],
    [0, 0, 0, 0, 0, -1]
])
test_check_fork(env, board, row=2, col=5, player=1, expected_output=False)

# Test Case 3: Blocked Line
board = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [-1, 1, 1, 0, -1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0]
])
test_check_fork(env, board, row=2, col=3, player=1, expected_output=False)

# Test Case 4: Diagonal Fork
board = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, -1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, -1]
])
test_check_fork(env, board, row=3, col=3, player=1, expected_output=False)

board = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, -1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, -1]
])
test_check_fork(env, board, row=3, col=3, player=1, expected_output=False)