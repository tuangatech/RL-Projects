class Config:
    # Game parameters
    BOARD_SIZE = 6
    WIN_LENGTH = 4
    REWARD_WIN = 1.0                # Score for winning the game
    REWARD_DRAW = 0.0               # Score for drawing the game
    REWARD_FORK = 0.3               # Score for a fork (2 threats at once)
    REWARD_INVALID_MOVE = -1.0      # Score for invalid move

    # The relative values are important: Live 3 > Semi-Open 3 > Live 2
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
    EPISODES = 20000                # Number of training episodes 10000
    BATCH_SIZE = 96                 # Batch size for training 64
    REPLAY_CAPACITY = 10000         # Replay buffer capacity 10000 transitions
    TRAIN_START = 1000              # Start training after this many samples in the buffer
    TARGET_UPDATE = 300             # Update target network every TARGET_UPDATE episodes 100
    PLAY_AGENT_FREQ = EPISODES // 5   # Play against the bot every PLAY_BOT_FREQ episodes to monitor performance
    
    # Neural network parameters
    LEARNING_RATE = 1e-4            # Learning rate for the optimizer
    MIN_LEARNING_RATE = 2e-6        # Minimum learning rate for the optimizer
    LR_DECAY_START_FRAME = 10000    # Frame at which to start decaying the learning rate
    TOTAL_FRAMES = EPISODES * 30    # A frame is an interactions with the environment (transition). Est 30 frames per episode
    
    # RL parameters
    GAMMA = 0.99                    # Discount factor for future rewards
    EPSILON_START = 1.0             # Initial exploration rate
    EPSILON_MIN = 0.1               # Minimum exploration rate
    EPSILON_DECAY = 0.9999          # Decay rate for exploration
    