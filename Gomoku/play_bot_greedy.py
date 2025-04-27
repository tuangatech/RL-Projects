import torch
from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
from agents import GreedyAgent, RandomAgent
import numpy as np
from tqdm import tqdm
from game_config import Config

def evaluate(dqn_agent, dqn_first=True, num_games=50):
    env = GomokuEnv(board_size=Config.BOARD_SIZE, win_length=Config.WIN_LENGTH)
    target_agent = RandomAgent(board_size=6)

    results = {"dqn_win": 0, "agent_win": 0, "draw": 0}

    for i in tqdm(range(num_games)):
        obs = env.reset()
        done = False

        while not done:
            if (env.current_player == 1 and dqn_first) or (env.current_player == -1 and not dqn_first):
                action = dqn_agent.act(obs)
            else:
                action = target_agent.act(obs)

            obs, reward, done, info = env.step(action)

        # At the end of step(action), player was witched self.current_player *= -1
        # When dqn_first=True, the DQN agent plays as Player 1 (current_player == 1)
        # When dqn_first=False, the DQN agent plays as Player 2 (current_player == -1)
        if reward == 1:
            # env.current_player == -1 means the last move was made by the DQN agent
            # env.current_player == 1 means the last move was made by the target agent
            if (env.current_player == -1 and dqn_first) or (env.current_player == 1 and not dqn_first):
                results["dqn_win"] += 1
            else:
                results["agent_win"] += 1
        else:
            results["draw"] += 1

        if (i + 1) % 10 == 0:
            print(f"After {i + 1} games: {results}")

    return results


if __name__ == "__main__":    
    dqn_agent = DQNAgent(board_size=Config.BOARD_SIZE)
    dqn_agent.epsilon = 0.0  # no exploration during eval
    dqn_agent.q_net.load_state_dict(torch.load("dqn_gomoku.pt"))
    dqn_agent.q_net.eval()  # set the model to evaluation mode
    print("Model loaded dqn_gomoku.pt")

    print("DQN plays first:")
    results1 = evaluate(dqn_agent, dqn_first=True)
    print(results1)

    print("\nGreedy plays first:")
    results2 = evaluate(dqn_agent, dqn_first=False)
    print(results2)

    print("\nTotal Results:")
    total = {
        "DQN Wins": results1["dqn_win"] + results2["dqn_win"],
        "Agent Wins": results1["agent_win"] + results2["agent_win"],
        "Draws": results1["draw"] + results2["draw"]
    }
    print(total)
