import random
import time
from gomoku_env import GomokuEnv

# Create environment
env = GomokuEnv()

# Run a single episode
obs = env.reset()
done = False
total_reward = 0

print("Starting a new game...\n")
env.render()

while not done:
    # Mask invalid moves
    flat_board = (obs[0] + obs[1]).reshape(-1)
    valid_actions = [i for i, val in enumerate(flat_board) if val == 0]
    action = random.choice(valid_actions)

    obs, reward, done, info = env.step(action)

    env.render()
    print(f"Player {env.current_player * -1} placed at {divmod(action, env.board_size)}")
    print(f"Reward: {reward:.2f} | Done: {done} | Info: {info}\n")
    total_reward += reward

    time.sleep(0.5)

print(f"Game over. Final reward: {total_reward}")
