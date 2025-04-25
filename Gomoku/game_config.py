class Config:
    # Game parameters
    BOARD_SIZE = 6
    WIN_LENGTH = 4
    REWARD_LENGTH2 = 0.03           # Score for 2 stones in a row, 2 open ends
    REWARD_LENGTH3 = 0.1            # Score for 3 stones in a row, 1 open end
    REWARD_LENGTH4 = 0.4            # Score for 3 stones in a row, 2 open ends
    REWARD_FORK = 0.3               # Score for a fork (2 threats at once)
    REWARD_PROXIMITY = 0.01         # Score for proximity to opponent's stones
    REWARD_WIN = 1.0                # Score for winning the game
    REWARD_DRAW = 0.0               # Score for drawing the game
    IMPACT_RADIUS = 3               # Consider 1 move impacts 3x3 area around it, so clear cache in that 3x3 area

    # Training parameters
    EPISODES = 50000                # Number of training episodes 10000
    BATCH_SIZE = 64                 # Batch size for training 64
    REPLAY_CAPACITY = 10000         # Replay buffer capacity 10000 transitions
    TRAIN_START = 1000              # Start training after this many samples in the buffer
    TARGET_UPDATE = 100             # Update target network every TARGET_UPDATE episodes
    
    # Neural network parameters
    LEARNING_RATE = 1e-4            # Learning rate for the optimizer
    MIN_LEARNING_RATE = 2e-6        # Minimum learning rate for the optimizer
    LR_DECAY_START_FRAME = 10000    # Frame at which to start decaying the learning rate
    TOTAL_FRAMES = EPISODES * 30    # A frame is an interactions with the environment (transition). Est 30 frames per episode
    
    # RL parameters
    GAMMA = 0.99                    # Discount factor for future rewards
    EPSILON_START = 1.0             # Initial exploration rate
    EPSILON_MIN = 0.1               # Minimum exploration rate
    EPSILON_DECAY = 0.99995         # Decay rate for exploration
    