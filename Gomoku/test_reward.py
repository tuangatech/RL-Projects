from gomoku_env import GomokuEnv
from game_config import Config
import numpy as np
# import pytest

def test_reward_shaping(env, board, action, expected_reward):
    """
    Test the reward shaping logic for a given board state and action.
    :param env: GomokuEnv instance.
    :param board: Initial board state as a NumPy array.
    :param action: Action to take (flattened index).
    :param expected_reward: Expected reward after taking the action.
    """
    env.reset()  # Reset the environment
    env.board = board  # Set the initial board state
    env.current_player = 1  # Set current player to 1 for testing
    row, col = divmod(action, env.board_size)
    
    # Calculate opponent's threat before the move
    # opponent = -env.current_player
    # opponent_threat_before = env._calculate_global_threat(opponent)
    # player_threat_before = env._calculate_global_threat(env.current_player)

    print("Initial Board:")
    print(board)
    print(f"Taking action at (row={row}, col={col})")

    # Take the action ^^^^^^^^^^^^^
    next_state, reward, done, info = env.step(action)
    print(f"Reward after action: {reward}\n")
    
    print("Board After Move:")
    if not done:
        print(next_state[1].astype(np.int32) - next_state[0].astype(np.int32))
    else: # Game over, no switching turns
        print(next_state[0].astype(np.int32) - next_state[1].astype(np.int32))
    print(f"Expected Reward: {expected_reward}, Actual Reward: {reward}")
    assert abs(reward - expected_reward) < 1e-2, "Test FAILED !!!\n"
    print("Test PASSED!\n")

# Initialize the environment
env = GomokuEnv(board_size=Config.BOARD_SIZE, win_length=Config.WIN_LENGTH)

# Test Case 1: Threat Evaluation
board1 = np.array([
    [ 0,  0,  0, 0, 0, 0],
    [ 0, -1,  0, 0, 0, 0],
    [-1,  0,  0, 1, 1, 0],
    [ 0,  0, -1, 0, 0, 0],
    [ 1,  0, -1, 0, 0, 0],
    [ 0,  0,  0, 0, 0, 0]
])
player_before = Config.REWARD_LIVE2
player_after =  Config.REWARD_SEMI_OPEN3
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = Config.REWARD_LIVE2
opponent_after = Config.REWARD_LIVE2
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = 0 #Config.REWARD_FORK
# soon_win = Config.REWARD_SOON_WIN
# block_soon_win = 0
expected_reward = (player_after - player_before) + (opponent_before - opponent_after) + fork #+ soon_win + block_soon_win
test_reward_shaping(env, board1, action=17, expected_reward=expected_reward)

# Test Case 2: Fork Detection
board2 = np.array([
    [ 0,  0,  0, 0, 0, 0],
    [ 0, -1,  0, 0, 0, 0],
    [-1,  1, -1, 1, 0, 0],
    [ 0,  1,  1, 0, 0, 0],
    [ 0,  0,  0, 0, 0, 0],
    [-1,  0,  0, 0, 0, 0]
])
player_before = 3*Config.REWARD_LIVE2
player_after = 2*Config.REWARD_SEMI_OPEN3 + 2*Config.REWARD_LIVE2
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = Config.REWARD_LIVE2
opponent_after = Config.REWARD_LIVE2
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = Config.REWARD_FORK
# soon_win = Config.REWARD_SOON_WIN
# block_soon_win = 0
expected_reward = player_after - player_before + (opponent_before - opponent_after) + fork #+ soon_win + block_soon_win
test_reward_shaping(env, board2, action=25, expected_reward=expected_reward)

# Test Case 3: Blocking Opponent's soon win
board3 = np.array([
    [0, 0, 0, 0,  1, 0],
    [0, 0, 0, 0, -1, 0],
    [0, 1, 0, 1,  0, 0],
    [0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0,  0, 0]
])
player_before = 0
player_after = Config.REWARD_LIVE2 + Config.REWARD_GAP4
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = Config.REWARD_LIVE2 + Config.REWARD_GAP4
opponent_after = 0
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = 0
# soon_win = Config.REWARD_SOON_WIN
# block_soon_win = Config.REWARD_BLOCK_SOON_WIN
expected_reward = player_after - player_before + (opponent_before - opponent_after) + fork #+ soon_win + block_soon_win
test_reward_shaping(env, board3, action=16, expected_reward=expected_reward)

obs = env._get_obs()
print(f"player map \n{obs[0]}")
print(f"opponent map \n{obs[1]}")

# Test Case 4: Strategic Defense
board4 = np.array([
    [0, 0,  0,  0, 0, 0],
    [0, 1,  0,  0, 0, 0],
    [0, 0, -1, -1, 0, 0],
    [0, 0,  0,  0, 0, 0],
    [0, 0,  0, -1, 0, 0],
    [0, 0,  0,  0, 0, 0]
])
player_before = 0
player_after = Config.REWARD_LIVE2
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = Config.REWARD_LIVE2
opponent_after = 0
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = 0
# soon_win = 0
# block_soon_win = 0
expected_reward = player_after - player_before + (opponent_before - opponent_after) + fork #+ soon_win + block_soon_win
test_reward_shaping(env, board4, action=13, expected_reward=expected_reward)

# Test Case 5: Full Board Evaluation
board5 = np.array([
    [1,  1,  0, -1, -1,  0],
    [0,  0,  1,  0,  0, -1],
    [0, -1,  1,  0,  0,  0],
    [0,  0,  1, -1,  0,  0],
    [0,  0, -1,  0,  1,  0],
    [0,  0,  0,  0,  0, -1]
])
player_before = Config.REWARD_SEMI_OPEN3
player_after = Config.REWARD_WIN
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = Config.REWARD_LIVE2 + Config.REWARD_LIVE2
opponent_after = Config.REWARD_LIVE2
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = 0
# soon_win = 0
# block_soon_win = 0
expected_reward = player_after - player_before + (opponent_before - opponent_after) + fork #+ soon_win + block_soon_win
test_reward_shaping(env, board5, action=2, expected_reward=Config.REWARD_WIN)

# Test Case 6: Fork
board6 = np.array([
    [1,  1,  0,  0, -1,  0],
    [0,  0,  1, -1,  0, -1],
    [0, -1,  1,  0, -1,  0],
    [0,  0,  0,  0,  0, -1],
    [0,  0, -1,  0,  1,  0],
    [0,  0,  0,  0,  0, -1]
])
player_before = Config.REWARD_LIVE2
player_after = 2*Config.REWARD_SEMI_OPEN3
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = Config.REWARD_SEMI_OPEN3 + Config.REWARD_GAP4
opponent_after = Config.REWARD_GAP4
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = Config.REWARD_FORK
# soon_win = Config.REWARD_SOON_WIN
# block_soon_win = Config.REWARD_BLOCK_SOON_WIN
expected_reward = (player_after - player_before) + (opponent_before - opponent_after) + fork #+ soon_win + block_soon_win
test_reward_shaping(env, board6, action=2, expected_reward=expected_reward)

# Test Case 7: Passive Move
board7 = np.array([
    [0, 0,  0,  0, 0, 0],
    [0, 1,  0,  0, 0, 0],
    [0, 0, -1, -1, 0, 0],
    [0, 0,  0,  0, 0, 0],
    [0, 0,  0, -1, 0, 0],
    [0, 0,  0,  0, 0, 0]
])
player_before = 0
player_after = 0
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = Config.REWARD_LIVE2
opponent_after = Config.REWARD_LIVE2
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = 0
# soon_win = 0
# block_soon_win = 0
passive = Config.REWARD_PASSIVE
expected_reward = (player_after - player_before) + (opponent_before - opponent_after) + fork + passive #+ soon_win + block_soon_win
test_reward_shaping(env, board7, action=4, expected_reward=expected_reward)

# Test Case 8: Sequences in the same line
board1 = np.array([
    [ 0,  0,  0, 0, 0, 1],
    [-1, -1, -1, 0, 1, 0],
    [ 0,  0,  0, 0, 0, 0],
    [ 0,  0,  1, 0, 0, 0],
    [ 0,  1,  0, 0, 0, 0],
    [ 1,  0,  0, 0, 0, 0]
])
player_before = Config.REWARD_SEMI_OPEN3 + Config.REWARD_GAP4 + Config.REWARD_GAP4
player_after = Config.REWARD_WIN
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = Config.REWARD_SEMI_OPEN3
opponent_after = Config.REWARD_SEMI_OPEN3
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = 0
# soon_win = 0
# block_soon_win = 0
expected_reward = (player_after - player_before) + (opponent_before - opponent_after) + fork #+ soon_win + block_soon_win
test_reward_shaping(env, board1, action=15, expected_reward=Config.REWARD_WIN)

board3 = np.array([
    [ 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0]
])

board3 = np.array([
    [ 0, 0, 0, 0, 0, 0],
    [-1, 1, 1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0],
    [ 1, 1, 0, 1, 1,-1],
    [ 0, 0, 0, 0, 0, 0]
])
player_before = Config.REWARD_GAP4 + Config.REWARD_GAP4 + Config.REWARD_GAP4
player_after = Config.REWARD_GAP4 + Config.REWARD_GAP4 + Config.REWARD_GAP4 + Config.REWARD_SEMI_OPEN3
print(f"Player_threat_delta: {player_after - player_before}")
opponent_before = 0
opponent_after = 0
print(f"Opponent_threat_delta: {opponent_before - opponent_after}")
fork = 0
# soon_win = Config.REWARD_SOON_WIN
# block_soon_win = 0
passive = 0
expected_reward = player_after - player_before + (opponent_before - opponent_after) + fork + passive #+ soon_win + block_soon_win
test_reward_shaping(env, board3, action=9, expected_reward=expected_reward)