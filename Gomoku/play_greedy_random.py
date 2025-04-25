import numpy as np
from gomoku_env import GomokuEnv
from agents import GreedyAgent, RandomAgent


def evaluate_greedy_vs_random(num_games=10, greedy_first=True):
    env = GomokuEnv(board_size=6, win_length=4)
    greedy_agent = GreedyAgent(board_size=6)
    random_agent = RandomAgent(board_size=6)

    results = {"greedy_win": 0, "random_win": 0, "draw": 0}

    for _ in range(num_games):
        obs = env.reset()
        done = False

        while not done:
            # Greedy agent's turn
            # env.current_player == 1 means the first player
            # env.current_player == -1 means the second player
            if (env.current_player == 1 and greedy_first) or (env.current_player == -1 and not greedy_first):
                action = greedy_agent.act(obs)
            # Random agent's turn
            else:
                action = random_agent.act(obs)

            obs, reward, done, info = env.step(action)


        if reward == 1:
            # env.current_player represents the player who made the winning move (last move)
            # because no player alternation after env._check_win()
            # (env.current_player == 1 and greedy_first) means greedy plays first and the last move was made by greedy
            # (env.current_player == -1 and not greedy_first) means random plays first and the last move was made by greedy
            winner = "greedy_win" if (env.current_player == 1 and greedy_first) or \
                (env.current_player == -1 and not greedy_first) else "random_win"
            results[winner] += 1
        else:
            results["draw"] += 1

    return results

if __name__ == "__main__":
    print("Greedy goes first:")
    results1 = evaluate_greedy_vs_random(num_games=10, greedy_first=True)
    print(results1)

    print("\nRandom goes first:")
    results2 = evaluate_greedy_vs_random(num_games=10, greedy_first=False)
    print(results2)

    total = {
        "Greedy Wins": results1["greedy_win"] + results2["greedy_win"],
        "Random Wins": results1["random_win"] + results2["random_win"],
        "Draws": results1["draw"] + results2["draw"]
    }
    print("\nTotal Results:")
    print(total)
