class Config:
    # Game parameters
    BOARD_SIZE = 6
    WIN_LENGTH = 4
    REWARD_WIN = 1.0                # Score for winning the game
    REWARD_DRAW = 0.0               # Score for drawing the game
    REWARD_FORK = 0.3               # Score for a fork (2 threats at once)
    REWARD_INVALID_MOVE = -1.0      # Score for invalid move
    # REWARD_LENGTH2 = 0.03           # Score for 2 stones in a row, 2 open ends
    # REWARD_LENGTH3 = 0.1            # Score for 3 stones in a row, 1 open end
    # REWARD_LENGTH4 = 0.4            # Score for 3 stones in a row, 2 open ends
    # REWARD_PROXIMITY = 0.01         # Score for proximity to opponent's stones
    # IMPACT_RADIUS = 3               # Consider 1 move impacts 3x3 area around it, so clear cache in that 3x3 area

    # Define scores for different patterns - these are heuristic values, tune them!
    # The values represent the contribution of finding *one instance* of this pattern.
    # The relative values are important: Live 3 > Semi-Open 4 > Semi-Open 3 > Live 2
    REWARD_LIVE2 = 0.03       # -XX-
    REWARD_SEMI_OPEN3 = 0.1   # -XXXo or oXXX-
    REWARD_LIVE3 = 0.4        # -XXX- (Strong threat!)
    # SCORE_SEMI_OPEN4 = 1.0   # -XXXXo or oXXXX- (Immediate winning threat on next move)
    # (Live 4 -XXXX- is game ending, handled by win condition)

    # Patterns to check and their scores
    # Using a dictionary for easy lookup
    PATTERN_SCORES = {
        "-XX-": REWARD_LIVE2,
        "-XXXo": REWARD_SEMI_OPEN3,
        "oXXX-": REWARD_SEMI_OPEN3,
        "-XXX-": REWARD_LIVE3,
        # "-XXXXo": SCORE_SEMI_OPEN4,
        # "oXXXX-": SCORE_SEMI_OPEN4,
    }
    # Maximum length of sequence needed to check for patterns, which are SEMI-OPEN3 and LIVE3
    CHECK_SEQUENCE_LENGTH = 5 # 5 is the maximum length of PATTERN_SCORES keys

    # Training parameters
    EPISODES = 30000                # Number of training episodes 10000
    BATCH_SIZE = 128                 # Batch size for training 64
    REPLAY_CAPACITY = 10000         # Replay buffer capacity 10000 transitions
    TRAIN_START = 1000              # Start training after this many samples in the buffer
    TARGET_UPDATE = 300             # Update target network every TARGET_UPDATE episodes 100
    
    # Neural network parameters
    LEARNING_RATE = 1e-4            # Learning rate for the optimizer
    MIN_LEARNING_RATE = 2e-6        # Minimum learning rate for the optimizer
    LR_DECAY_START_FRAME = 10000    # Frame at which to start decaying the learning rate
    TOTAL_FRAMES = EPISODES * 30    # A frame is an interactions with the environment (transition). Est 30 frames per episode
    
    # RL parameters
    GAMMA = 0.99                    # Discount factor for future rewards
    EPSILON_START = 1.0             # Initial exploration rate
    EPSILON_MIN = 0.05              # Minimum exploration rate
    EPSILON_DECAY = 0.99995         # Decay rate for exploration
    