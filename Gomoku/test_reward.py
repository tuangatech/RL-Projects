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
    
    # # Calculate opponent's threat after the move
    # opponent_threat_after = env._calculate_global_threat(opponent)
    # player_threat_after = env._calculate_global_threat(-env.current_player)
    
    # # Add strategic defense reward
    # est_reward += opponent_threat_before - opponent_threat_after
    # est_reward += player_threat_after - player_threat_before
    # if env._check_fork(row, col, -env.current_player):
    #     reward += Config.REWARD_FORK
    
    # print(f"Opponent's Threat Before: {opponent_threat_before}, After: {opponent_threat_after}")
    # print(f"Player's Threat Before: {player_threat_before}, After: {player_threat_after}")
    # print(f"Fork Detected: {env._check_fork(row, col, env.current_player)}")

    # print(f"Board 1:\n{next_state[0]}")
    # print(f"Board 2:\n{next_state[1]}\n")
    print("Board After Move:")
    print(next_state[1].astype(np.int32) - next_state[0].astype(np.int32))   
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
    [ 0,  1, -1, 0, 0, 0],
    [ 1,  0, -1, 0, 0, 0],
    [ 0,  0,  0, 0, 0, 0]
])
player_before = Config.REWARD_LIVE2
player_after = Config.REWARD_LIVE3 + Config.REWARD_SEMI_OPEN3
opponent_before = Config.REWARD_LIVE2
opponent_after = 0
fork = Config.REWARD_FORK
expected_reward = (player_after - player_before) + (opponent_before - opponent_after)  + fork
test_reward_shaping(env, board1, action=14, expected_reward=expected_reward)

# Test Case 2: Fork Detection
board2 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [-1, 1, -1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0]
])
player_before = 4*Config.REWARD_LIVE2
player_after = 2*Config.REWARD_SEMI_OPEN3 + 2*Config.REWARD_LIVE2
opponent_before = Config.REWARD_LIVE2
opponent_after = Config.REWARD_LIVE2
fork = Config.REWARD_FORK
expected_reward=player_after - player_before + (opponent_before - opponent_after)  + fork
# test_reward_shaping(env, board2, action=25, expected_reward=expected_reward)

# Test Case 3: Proximity Rewards
board3 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0]
])
player_before = 0
player_after = Config.REWARD_LIVE2
opponent_before = Config.REWARD_LIVE2
opponent_after = Config.REWARD_LIVE2
fork = 0
expected_reward = player_after - player_before + (opponent_before - opponent_after)  + fork
# test_reward_shaping(env, board3, action=13, expected_reward=expected_reward)

# Test Case 4: Strategic Defense
board4 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, -1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0]
])
player_before = 0
player_after = Config.REWARD_LIVE2
opponent_before = Config.REWARD_LIVE2
opponent_after = 0
fork = 0
expected_reward = player_after - player_before + (opponent_before - opponent_after)  + fork
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
opponent_before = Config.REWARD_LIVE2 + Config.REWARD_LIVE2
opponent_after = Config.REWARD_LIVE2
fork = 0
expected_reward = player_after - player_before + (opponent_before - opponent_after) + fork
test_reward_shaping(env, board5, action=2, expected_reward=Config.REWARD_WIN)