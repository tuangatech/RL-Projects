import pygame
import numpy as np
from gomoku_env import GomokuEnv
from agents import GreedyAgent
from game_config import Config

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
CELL_SIZE = SCREEN_WIDTH // Config.BOARD_SIZE
LINE_COLOR = (50, 50, 50)  # Dark gray for grid lines
MARGIN = 30  # Margin for the board
PLAYER_COLORS = {
    1: (200, 0, 0),  # Red for Player 1 (Human)
    -1: (0, 100, 200)  # Blue for Player 2 (Greedy Agent)
}
BACKGROUND_COLOR = (240, 240, 240)  # White background

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Gomoku with GreedyAgent")

# Draw the Gomoku board
def draw_board():
    screen.fill(BACKGROUND_COLOR)
    for i in range(1, Config.BOARD_SIZE):
        # Vertical lines
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_HEIGHT), 2)
        # Horizontal lines
        pygame.draw.line(screen, LINE_COLOR, (0, i * CELL_SIZE), (SCREEN_WIDTH, i * CELL_SIZE), 2)
    pygame.display.flip()

# Draw a stone on the board
def draw_stone(row, col, player):
    center = (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2)
    radius = CELL_SIZE // 4
    pygame.draw.circle(screen, PLAYER_COLORS[player], center, radius)
    pygame.display.flip()

# Convert mouse click position to grid coordinates
def get_grid_position(mouse_x, mouse_y):
    row = mouse_y // CELL_SIZE
    col = mouse_x // CELL_SIZE
    return row, col

# Main game loop
def play_game():
    env = GomokuEnv(board_size=Config.BOARD_SIZE, win_length=Config.WIN_LENGTH)
    agent = GreedyAgent(board_size=Config.BOARD_SIZE)

    running = True
    human_player = 1  # Human plays as Player 1
    draw_board()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and env.current_player == human_player:
                # Human player's turn
                mouse_x, mouse_y = event.pos
                row, col = get_grid_position(mouse_x, mouse_y)
                action = row * Config.BOARD_SIZE + col

                # Check if the move is valid
                if env.board[row, col] != 0:  # Cell is already occupied
                    continue

                # Make the move
                obs, reward, done, info = env.step(action)
                draw_stone(row, col, human_player)

                # Check if the game is over
                if done:
                    if reward == 1:
                        print("Human wins!")
                    else:
                        print("Draw!")
                    running = False
                    break

                # Greedy Agent's turn
                agent_action = agent.act(obs)
                agent_row, agent_col = divmod(agent_action, Config.BOARD_SIZE)
                obs, reward, done, info = env.step(agent_action)
                draw_stone(agent_row, agent_col, -human_player)

                # Check if the game is over
                if done:
                    if reward == 1:
                        print("Greedy Agent wins!")
                    else:
                        print("Draw!")
                    running = False
                    break

    pygame.quit()

if __name__ == "__main__":
    play_game()
