class Config:
    # Game parameters
    BOARD_SIZE          = 6
    WIN_LENGTH          = 4
    REWARD_WIN          = 1.0                # Score for winning the game
    REWARD_DRAW         = -0.05              # Score for drawing the game
    REWARD_INVALID_MOVE = -1.0      # Score for invalid move

    # The relative values are important: Live 3 > Semi-Open 3 > Live 2
    REWARD_LIVE2        = 0.003            # -XX-
    REWARD_SEMI_OPEN3   = 0.01        # -XXXo or oXXX-
    REWARD_LIVE3        = 0.03              # -XXX- (Strong threat!)
    REWARD_GAP4         = 0.01               # XX-X (Strong threat!)
    REWARD_FORK         = 0.03               # Score for a fork (2 threats at once)
    REWARD_PASSIVE      = -0.003      # Score for passive moves (not threatening, not blocking opponent)
    # REWARD_SOON_WIN = 0.05           # The move that will lead to a win in the next turn - additional score
    # REWARD_BLOCK_SOON_WIN = 0.05     # The move that blocks the opponent's winning move - additional score

    # Patterns to check and their scores, X=player, o=opponent, -=empty
    # The patterns are defined as strings of length 5, where:
    PATTERN_SCORES = {
        "-XX-": REWARD_LIVE2,
        "XXX-": REWARD_SEMI_OPEN3,
        "-XXX": REWARD_SEMI_OPEN3,
        "-XXX-": REWARD_LIVE3,
        "XX-X": REWARD_GAP4,
        "X-XX": REWARD_GAP4,
        # "-XX-X": REWARD_GAP4,
        # "XX-X-": REWARD_GAP4,
        # "oXX-X": REWARD_GAP4,
        # "XX-Xo": REWARD_GAP4,
        # "-X-XX": REWARD_GAP4,
        # "X-XX-": REWARD_GAP4,
        # "X-XXo": REWARD_GAP4,
        # "oX-XX": REWARD_GAP4,
        "XX-XX": REWARD_GAP4,
    }
    # Maximum length of sequence needed to check for patterns, which are SEMI-OPEN3 and LIVE3
    CHECK_SEQUENCE_LENGTH = 5 # 5 is the maximum length of PATTERN_SCORES keys

    # Training parameters
    EPISODES = 7000                 # -- Number of training episodes 10000
    BATCH_SIZE = 64                 # Batch size for training 64
    REPLAY_CAPACITY = 50000         # Replay buffer capacity 10000 transitions
    TRAIN_START = 2000             # -- Start training after this many transitions in the buffer. 
                                    # Epsilon is decayed after this point. 10000 transitions ~ 430 episodes
    # TARGET_UPDATE = 200             # Update target network every TARGET_UPDATE episodes 100
    PLAY_AGENT_FREQ = EPISODES // 30 # Play against the bot every PLAY_BOT_FREQ episodes to monitor performance
    GAMES_AGAINST_AGENT = 20        # Number of games to play against the bot for evaluation
    
    # Neural network parameters
    LEARNING_RATE = 2e-5            # Learning rate for the optimizer
    MIN_LEARNING_RATE = 2e-6        # Minimum learning rate for the optimizer
    LR_DECAY_START_FRAME = TRAIN_START + 2000     # Frame at which to start decaying the learning rate
    TOTAL_FRAMES = EPISODES * 24    # A frame is an interactions with the environment (transition). Est 20 frames per episode
    
    # RL parameters
    GAMMA = 0.95                    # Discount factor for future rewards. A high gamma combined with the accumulating intermediate rewards makes it even more likely that the total expected discounted sum will exceed 1.0
    EPSILON_START = 1.0             # Initial exploration rate
    EPSILON_MIN = 0.1              # Minimum exploration rate
    EPSILON_DECAY = 0.9995         # -- Decay rate for exploration (30000 episodes: 0.9999, 15000 episodes: 0.9998, 
                                    # 10000 episodes: 0.9997, 5000 episodes: 0.9995)
    UPDATE_TAU = 0.01               # Soft update parameter for target network (0.01 is a common value)
    REWARD_SCALING_START = 1000     # Start scaling after this many episodes