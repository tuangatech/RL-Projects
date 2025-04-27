import pygame
import sys
import numpy as np
from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
import torch
from game_config import Config

# --- CONFIG ---
CELL_SIZE = 100
MARGIN = 30  # Increased margin for better spacing
WINDOW_SIZE = CELL_SIZE * Config.BOARD_SIZE + 2 * MARGIN
LINE_COLOR = (50, 50, 50)
BG_COLOR = (240, 240, 240)  # Light gray background for contrast
X_COLOR = (200, 0, 0)  # Red for X
O_COLOR = (0, 100, 200)  # Blue for O

# --- INIT ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Gomoku: Human (O) vs Agent (X)")
font = pygame.font.SysFont(None, 48)

# Load and set the custom icon
icon = pygame.image.load("icon.png")
pygame.display.set_icon(icon)

# --- LOAD ENV & AGENT ---
env = GomokuEnv(board_size=Config.BOARD_SIZE, win_length=Config.WIN_LENGTH)
obs = env.reset()

agent = DQNAgent(board_size=Config.BOARD_SIZE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
agent.q_net.load_state_dict(torch.load("dqn_gomoku.pt", map_location=agent.device))
agent.q_net.eval()

# --- FUNCTIONS ---
def draw_board(board):
    screen.fill(BG_COLOR)

    # Draw vertical and horizontal lines
    for i in range(Config.BOARD_SIZE + 1):  # +1 ensures the right and bottom borders are drawn
        start_x = MARGIN
        start_y = MARGIN + i * CELL_SIZE
        end_x = MARGIN + Config.BOARD_SIZE * CELL_SIZE
        end_y = MARGIN + i * CELL_SIZE
        pygame.draw.line(screen, LINE_COLOR, (start_x, start_y), (end_x, start_y), 2)  # Horizontal line

        start_x = MARGIN + i * CELL_SIZE
        start_y = MARGIN
        end_x = MARGIN + i * CELL_SIZE
        end_y = MARGIN + Config.BOARD_SIZE * CELL_SIZE
        pygame.draw.line(screen, LINE_COLOR, (start_x, start_y), (start_x, end_y), 2)  # Vertical line

    # Draw stones
    for y in range(Config.BOARD_SIZE):
        for x in range(Config.BOARD_SIZE):
            cx = MARGIN + x * CELL_SIZE + CELL_SIZE // 2
            cy = MARGIN + y * CELL_SIZE + CELL_SIZE // 2
            if board[y, x] == 1:  # Agent's stone (X)
                pygame.draw.line(screen, X_COLOR, (cx - 25, cy - 25), (cx + 25, cy + 25), 6)
                pygame.draw.line(screen, X_COLOR, (cx + 25, cy - 25), (cx - 25, cy + 25), 6)
            elif board[y, x] == -1:  # Human's stone (O)
                pygame.draw.circle(screen, O_COLOR, (cx, cy), 25, 6)

    pygame.display.flip()

def show_message(text):
    msg = font.render(text, True, (0, 0, 0))
    rect = msg.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    screen.blit(msg, rect)
    pygame.display.flip()
    pygame.time.wait(2000)

def pos_to_action(pos):
    x = (pos[0] - MARGIN) // CELL_SIZE
    y = (pos[1] - MARGIN) // CELL_SIZE
    return y * Config.BOARD_SIZE + x

# --- GAME LOOP ---
running = True
obs = env.reset()

while running:
    draw_board(env.board)

    if env.current_player == -1:  # Human turn
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                action = pos_to_action(event.pos)
                if action in env.get_valid_actions():
                    obs, reward, done, info = env.step(action)
                    draw_board(env.board)
                    if done:
                        if reward == 1:
                            show_message("You Win!")
                        elif reward == -1:
                            show_message("Agent Wins!")
                        else:
                            show_message("Draw!")
                        obs = env.reset()
    else:  # Agent turn
        action = agent.act(obs, epsilon=0.0)
        obs, reward, done, info = env.step(action)
        draw_board(env.board)
        if done:
            if reward == 1:
                show_message("Agent Wins!")
            elif reward == -1:
                show_message("You Win!")
            else:
                show_message("Draw!")
            obs = env.reset()

pygame.quit()
sys.exit()