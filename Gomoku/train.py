import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime  # Add import for timestamp
from game_config import Config

# Init
env = GomokuEnv(Config.BOARD_SIZE, Config.WIN_LENGTH)
agent = DQNAgent(Config.BOARD_SIZE)
# buffer = PrioritizedReplayBuffer(Config.REPLAY_CAPACITY, alpha=0.6)  # Alpha controls prioritization strength
buffer = ReplayBuffer(Config.REPLAY_CAPACITY)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/gomoku_dqn_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

episode_losses = []

start_time = datetime.now()
print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

for episode in tqdm(range(1, Config.EPISODES + 1)): 
    state = env.reset()
    done = False
    trajectory = []

    # Play a game
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # Normalize reward for stable training
        reward = np.clip(reward, -1.0, 1.0)  # Clip rewards to [-1, 1]
        trajectory.append((state, action, reward, next_state, done))
        state = next_state

    # Push all transitions with correct rewards to the buffer
    for i, (s, a, r, s_, d) in enumerate(trajectory):        
        # Self-play trick: reward is always from current player's view
        # So we flip reward for the previous player
        # If it's the opponent's turn, flip reward
        if i % 2 == 1:
            r = -r
        buffer.push(s, a, r, s_, d)    

    # Train if enough samples in buffer
    # Sample a batch from the replay buffer and train the agent
    if len(buffer) > Config.TRAIN_START:
        batch = buffer.sample(Config.BATCH_SIZE)
        loss = agent.train_step(batch)
        episode_losses.append(loss)
        agent.decay_epsilon()

        # Log loss and epsilon
        writer.add_scalar("Loss/train", loss, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

    # Update target net
    if episode % Config.TARGET_UPDATE == 0:
        agent.update_target()

    # Logging every 500 episodes
    if episode % (Config.EPISODES // 20) == 0:  # 500
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Episode {episode} | Epsilon: {agent.epsilon:.3f} | Last Loss: {loss:.4f}")  # Buffer size: {len(buffer)} | 

print(f"Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time: {datetime.now() - start_time}")
print(f"Final epsilon: {agent.epsilon:.3f}")

# Close TensorBoard writer
writer.close()

# Save model
torch.save(agent.q_net.state_dict(), "dqn_gomoku.pt")
print("Model saved to dqn_gomoku.pt")

# ==========
# Plotting training curves
plt.figure(figsize=(15, 6))

# Loss plot
smoothed_losses = pd.Series(episode_losses).rolling(Config.EPISODES // 100).mean()
plt.plot(smoothed_losses, label='Training Loss')
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.xlabel("Training Step", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("DQN Loss Curve", fontsize=14)
plt.legend()

plt.tight_layout()
timestamp = datetime.now().strftime("%m-%d-%H-%M")
plt.savefig(f"runs/training_losses_{timestamp}_{Config.EPISODES}.png")
plt.close()
