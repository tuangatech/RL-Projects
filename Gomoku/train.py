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
from agents import GreedyAgent, RandomAgent
from game_config import Config

# Init
env = GomokuEnv(Config.BOARD_SIZE, Config.WIN_LENGTH)
agent = DQNAgent(Config.BOARD_SIZE)
# buffer = PrioritizedReplayBuffer(Config.REPLAY_CAPACITY, alpha=0.6)  # Alpha controls prioritization strength
buffer = ReplayBuffer(Config.REPLAY_CAPACITY)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/gomoku_dqn_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

episode_losses = []
episode_epsilons = []
win_rates = []  # To store win rates during evaluation

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

        if reward != Config.REWARD_WIN and reward != Config.REWARD_DRAW:
            # Scale down the reward as training progresses to reduce 
            # the influence of intermediate rewards, to encourage the agent to focus on winning
            reward_scale = max(0.2, 1 - (episode / Config.EPISODES))  
            reward *= reward_scale
        # There are might be many LIVE3 and FORK patterns in the game, 
        # so we need to clip the reward to avoid exploding gradients
        # Clip rewards to [-1, 1] to normalize reward for stable training
        reward = np.clip(reward, -1.0, 1.0)
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
        episode_epsilons.append(agent.epsilon)
        agent.decay_epsilon()

        # Log loss and epsilon
        writer.add_scalar("Loss/train", loss, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

    # Update target net
    if episode % Config.TARGET_UPDATE == 0:
        agent.update_target()

    # Periodic evaluation every 2000 episodes
    if episode % Config.PLAY_AGENT_FREQ == 0:
        target_agent = RandomAgent(board_size=Config.BOARD_SIZE)
        wins = 0
        for _ in range(Config.GAMES_AGAINST_AGENT):  # Play 20 games against a target agent
            state = env.reset()
            done = False
            player_placed_stone = 0
            while not done:
                if env.current_player == 1:  # DQN agent plays as Player 1
                    action = agent.act(state, epsilon=0.0)  # No exploration during evaluation
                else:  # Target agent plays as Player 2
                    action = target_agent.act(state)
                player_placed_stone = env.current_player
                state, reward, done, _ = env.step(action)
            # Check if the DQN agent won
            if reward == 1.0 and player_placed_stone == 1:  # DQN agent wins
                wins += 1
        win_rate = wins / Config.GAMES_AGAINST_AGENT
        win_rates.append(win_rate)
        print(f"Evaluating at episode {episode}: Win rate = {win_rate:.2f}")
        writer.add_scalar("Win Rate", win_rate, episode)
    
    # Logging
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
smoothed_losses = pd.Series(episode_losses).rolling(Config.EPISODES // 150).mean()
ax1 = plt.gca()  # Get the current axis
ax1.plot(smoothed_losses, label='Training Loss', color='tab:blue')
ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a secondary y-axis for epsilon and win rate
ax2 = ax1.twinx()
ax2.plot(episode_epsilons, label='Epsilon', color='orange', linestyle='--')
ax2.set_ylabel("Epsilon / Win Rate", fontsize=12, color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

# Bar plot for win rates
evaluation_episodes = list(range(Config.PLAY_AGENT_FREQ, Config.EPISODES + 1, Config.PLAY_AGENT_FREQ))  # Actual evaluation episodes
if len(win_rates) > 0:  # Ensure there are win rates to plot
    bar_positions = [x for x in evaluation_episodes[:len(win_rates)]]  # Align with win_rates length - 1000
    bars = ax2.bar(bar_positions, win_rates, width=Config.EPISODES // 90, color='gold', alpha=0.6, label='Win Rate')

    # Add values on top of each bar
    for bar, win_rate in zip(bars, win_rates):
        height = bar.get_height()  # Get the height of the bar
        ax2.text(
            bar.get_x() + bar.get_width() / 2,  # X position: center of the bar
            height + 0.02,                      # Y position: slightly above the bar
            f"{win_rate:.2f}",                  # Text: win rate rounded to 2 decimal places
            ha='center',                        # Horizontal alignment: center
            va='bottom',                        # Vertical alignment: bottom
            fontsize=10,                        # Font size
            color='black'                       # Text color
        )

# Title and legend
plt.title("DQN Loss Curve with Epsilon and Win Rate", fontsize=14)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
timestamp = datetime.now().strftime("%m-%d-%H-%M")
plt.savefig(f"runs/training_{timestamp}_{Config.EPISODES}.png")
plt.close()
