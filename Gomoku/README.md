# Gomoku AI with Deep Q-Learning

## Game
Gomoku with 6x6 board, the first to 4 consecutive stones (horizontal, vertical, diagonal) wins. Each action corresponds to placing a stone on a cell.

## Solution: DQN Agent

This solution trains an AI agent to play Gomoku 6,6,4 using a Deep Q-Network (DQN) approach enhanced with several modern reinforcement learning techniques. The training is performed via self-play. A Pygame interface for humans to play against the trained model.

**Expected Outcome**: Agent evolves from random moves to sophisticated strategies (e.g., forks, double threats) within 30,000 episodes.

### 1. Environment Design:

- A custom OpenAI Gym environment (GomokuEnv) models the game dynamics.
- It manages the 6x6 board state, player turns (1 and -1), valid moves, win/draw/invalid move detection.
- Smaller than standard 15×15 Gomoku to accelerate training while retaining strategic depth.

### 2. State Representation:

- The state provided to the agent is a 2-channel tensor of shape (`2, 6, 6`).
- Channel 1 represents the positions of the current player's stones (1 where stones exist, 0 otherwise).
- Channel 2 represents the positions of the opponent's stones.
- This spatial representation is suitable for processing by convolutional layers.

### 3. Network Architecture (Dueling DQN with CNN):

- A Convolutional Neural Network (CNN) serves as the function approximator for the Q-values. It uses `torch.nn`.
- The network takes the (2, 6, 6) state tensor as input.
- It employs convolutional layers (`nn.Conv2d`) with Batch Normalization (`nn.BatchNorm2d`) and `LeakyReLU` activation.
- Convolutional backbone (pattern extraction):
- `Conv(2→32, kernel=3×3, padding=1) → BatchNorm → LeakyReLU`
- `Conv(32→64, kernel=3×3, padding=1) → BatchNorm → LeakyReLU`
- __Why Convolutional networks?__ Use kernel to extract relevant spatial features and patterns from the board state (e.g., stone alignments).
- The architecture implements **Dueling DQN**. After the convolutional base, the network splits into two streams:
  - A Value Stream: Outputs a single scalar representing the value of the state, _V(s)_.
  - An Advantage Stream: Outputs a value for each possible action (36 actions), representing the advantage of taking that action compared to others in that state, _A(s,a)_.
- These streams are combined to produce the final Q-values for each action: Q(s,a)=V(s)+(A(s,a) - mean_A)

### 4. Learning Algorithm (Double DQN):

- The core learning uses Q-learning updates adapted for deep networks.
- **Experience Replay**: Transitions (`state, action, reward, next_state, done`) are stored in a `ReplayBuffer` of fixed capacity (`Config.REPLAY_CAPACITY`).
- Training Steps: During training, batches of experiences are randomly sampled from the buffer. This decorrelates updates and improves stability.
- Double DQN: To mitigate the overestimation bias common in standard DQN, the target Q-value calculation is modified:
  - The _online network_ (`q_net`) determines the best action (a*) to take in the `next_state`.
  - The _target network_ (`target_net`) evaluates the Q-value of taking that action a∗ in the `next_state`.
  - The target value becomes: T = reward + γ⋅Q_target(next_state, a∗)⋅(1−done).
- Loss Function: The Mean Squared Error (MSE) between the Q-value predicted by the online network for the action taken (`q_vals`) and the calculated target value (`targets`) is minimized using the Adam optimizer.
- Gradient Clipping: Gradient norms are clipped (`torch.nn.utils.clip_grad_norm_`) during backpropagation to prevent exploding gradients and further stabilize training.

### 5. Target Network & Soft Updates:

- A separate `target_net` with the same architecture as the `q_net` is used to provide stable targets during the Bellman update.
- Soft Updates: Instead of periodically copying weights, the `target_net` weights are updated slowly and smoothly after each training step by blending them with the `q_net` weights: `target_weights = tau * online_weights + (1 - tau) * target_weights`, where tau (Config.UPDATE_TAU) is a small constant (e.g., 0.005).

### 6. Training Strategy & Exploration:

- Self-Play: The agent learns by playing games against itself (its current policy), not fixed opponent, to adapt to improving strategies, avoiding overfitting to static adversaries. Experiences from both players' perspectives are stored in the replay buffer (with rewards appropriately flipped for the opponent's moves).
- Epsilon-Greedy Exploration: To balance exploration and exploitation, the agent uses an epsilon-greedy policy (`DQNAgent.act`). With probability `epsilon`, it chooses a random valid action; otherwise, it chooses the action with the highest Q-value.
- Exploration Decay: epsilon starts high (`Config.EPSILON_START`) and is gradually decayed over training episodes/frames (`Config.EPSILON_DECAY`) towards a minimum value (`Config.EPSILON_MIN`), shifting the agent from exploration to exploitation.
- Warm-up: Training updates only begin after a certain number of experiences (`Config.TRAIN_START`) have been collected in the replay buffer.
- Learning Rate Scheduling: A `LinearLR` scheduler dynamically adjusts the optimizer's learning rate during training, decreasing it from `Config.LEARNING_RATE` to `Config.MIN_LEARNING_RATE` starting after `Config.LR_DECAY_START_FRAME` frames, potentially improving convergence.
- Cyclical validation: Test vs. random agent every 1,000 episodes to track progress.

### 7. Reward Function (Terminal + Heuristic Shaping):

- Terminal Rewards: Clear rewards are given upon game termination: `Config.REWARD_WIN` (e.g., +1.0) for winning, `Config.REWARD_DRAW` (e.g., -0.1) for a draw, and `Config.REWARD_INVALID_MOVE` (e.g., -1.0) for attempting an illegal move.
- Action masking: occupied cells get Q(s,a) = −100 to avoid exploration waste.
- Heuristic / "Potential-Based" Reward Shaping: Dense, intermediate rewards are calculated at each step to potentially guide learning:
  - A "threat score" is computed based on Gomoku patterns (`Config.PATTERN_SCORES`) using the `_calculate_global_threat` helper function.
  - The step reward includes terms based on the change (delta) in the player's own threat score and the change in the opponent's threat score (rewarding increases in own threats and decreases in opponent threats, i.e., blocking). This resembles potential-based shaping but is calculated via heuristic pattern differences.
  - An additional reward (`Config.REWARD_FORK`) is given if the move creates a fork.
  - A small penalty (`Config.REWARD_PASSIVE`) is applied for moves that don't alter the threat landscape.
- _Caveat_: The effectiveness of complex reward shaping heavily depends on careful tuning of the relative magnitudes of heuristic vs. terminal rewards. If not balanced properly, it can lead to suboptimal policies focused on heuristics rather than winning. Simplifying this is often a key step in debugging.

### Improvement
1. **Monte Carlo Tree Search (MCTS)** : Combine reinforcement learning with MCTS to improve decision-making during gameplay.
2. Self-play training: periodically save checkpoints of the agent and **train against older versions** to simulate diverse opponents.
3. Consider using **prioritized experience replay** to prioritize important transitions (e.g., those leading to wins or losses). This can accelerate learning.

========
## Implementation

### 1. Environment
- Gym API
  - Inherit from `gym.Env`; define `observation_space = Box(0,2, shape=(2,6,6))` (one channel per player) and `action_space = Discrete(36)`.
- State & Move Validation
  - Board state: 0=empty, 1=agent’s stones, −1=opponent’s stones (or two binary channels).
  - On `step(a)`, reject invalid moves by masking them in the policy
- Terminal Check
  - After each move, scan rows, columns, and diagonals for 4 in a row → return `done=True` and reward.
  - If board full without 4 in a row, `done=True`, reward=0.


### 2. Reward Design
- Sparse “game‐end” reward:
  - +1 for win, –1 for loss, 0 for draw.
- Potential‐based reward shaping:
  - A line of 2 consecutive stones with 2 open ends receives a reward of 0.03
  - A line of 3 consecutive stones with 1 open end receives a reward of 0.1
  - A line of 3 consecutive stones with 2 open ends receives a reward of 0.4
  - A stone creating a fork of 2 lines of length-3, a fork has at least 1 open end, receives 0.3
  - A defensive reward is calculated by opponent's [threat after - threat before] the stone is placed

> **_NOTE:_** 
>
> In Gomoku, the default reward is sparse. This means the agent only receives feedback at the very end of the game. While simple, this approach can make learning slow because the agent has no intermediate signals to understand which actions are good or bad during the game.
>
> A potential function , denoted as Φ(s), assigns a numerical value to each state s based on some heuristic measure of progress or desirability. In Gomoku, the potential function is designed to capture how favorable the current board state is for the player.
> For example:
> Φ(s)=(# of open-ended 2’s)⋅α+(# of open-ended 3’s)⋅β
> - Open-ended 2’s : Lines of two consecutive stones with space to extend in both directions.
> - Open-ended 3’s : Lines of three consecutive stones with space to extend in both directions.
> - α and β: Weights that determine the importance of these patterns.
>
> If the agent creates an open-ended 3, the potential function increases, resulting in a positive shaping reward. Conversely, if the opponent creates a threat, the potential decreases, giving a negative shaping reward.
> 
> The potential function quantifies the "threat level" of the board state, rewarding the agent for creating opportunities to win and penalizing it for allowing the opponent to do so. Potential-based shaping uses the difference in potential between states (γ⋅Φ(s′)−Φ(s)) to provide intermediate rewards, preserving the optimal policy while alleviating reward sparsity.
> - s: Current state.
> - s′: Next state after taking an action.
> - γ: Discount factor (e.g., γ=0.99).


### 3. Agent Architecture
- Input: 2×6×6 tensor. `2` = 2 channels. First channel (6×6) : Represents the positions of the player's stones ("X" or "O"). Second channel (6×6) : Represents the opponent's stones. Each cell in the grid is either `1` (occupied by a stone) or `0` (empty).
- CNN vs. MLP
  - CNN strongly preferred: exploits spatial locality (ﬁlters detect lines, blocks, forks). In real life, much lower loss.
  - Example:
        ```
        Conv(2→32,3×3,pad=1) → ReLU → Conv(32→64,3×3,pad=1) → ReLU 
        → Flatten → FC(64*6*6 → 512) → ReLU → FC(512 → 36) 
        ```
  - Optionally use Dueling DQN: split into Value and Advantage streams after flattening.
- Output: Q‐value for each of 36 positions; mask illegal moves by setting their Q to −∞ before action selection.

> **_NOTE:_**
> - A Convolutional Neural Network (CNN) is ideal because it can detect these patterns efficiently using filters that slide over the board.
> - CNNs automatically learn features like "lines of 2," "open-ended 3's," and other strategic patterns through convolutional filters.
> - Feature maps are the outputs of convolutional layers, representing learned patterns or structures in the input data (e.g., horizontal lines of two stones, vertical threats). By using 32 feature maps , the network can learn 32 different types of patterns simultaneously. Each feature map focuses on detecting a specific type of pattern, allowing the network to build a rich understanding of the board state.
> - Progressive Learning: The first convolutional layer (with 32 feature maps) is typically responsible for learning low-level features like edges, simple lines, or isolated stones. The second convolutional layer with 64 feature maps can combine these low-level features to learn higher-level patterns like forks, blocks, or strategic opportunities.
> - Kernel: The kernel is a 3×3 filter that slides over the input. It performs element-wise multiplication and summation at each position. Since there are 2 input channels , the kernel also has 2 slices (one for each channel). These slices are combined to produce a single value for each position on the grid.
> - Batch normalization stabilizes training by normalizing the inputs to each layer. This can speed up convergence and improve performance.
> - Dropout for Regularization: Dropout can reduce overfitting by randomly zeroing out neurons during training. Dropout prevents the network from relying too heavily on specific neurons, improving generalization.
> - Output: The output is a 32×6×6 tensor. 32 feature maps each corresponds to a different filter. 6×6 grid : The spatial dimensions are preserved due to padding (pad=1).
> - Filter and Feature: When a filter is applied to the input, it produces a feature map highlighting positions where such patterns exist.
> - Flatten : Converts the 64×6×6 feature maps into a 1D vector of size 2304 (64 × 6 × 6).
> - FC(64\*6\*6 → 512): Fully connected layer to process the flattened features and output 512 neurons.
> - FC(512 → 36) : Final fully connected layer to output Q-values for all 36 positions on the board.
> - Dueling DQN splits the Q-value estimation into two streams:
>   - Value Stream (V) : Estimates the overall value of the current state (how good the state is regardless of the action).
>   - Advantage Stream (A) : Estimates the advantage of each action relative to others.
> - The output of the network is a vector of **Q-values**, one for each of the 36 positions on the board.
> - Some positions on the board may already be occupied, making them illegal moves. To handle this, mask illegal moves by setting their Q-values to −∞ before selecting an action. This ensures the agent never chooses an invalid position.
> - During **training**, use an **epsilon-greedy** policy to balance exploration and exploitation. During **evaluation**, select the action with the highest Q-value after masking illegal moves.
> - **Dueling DQN** architect is an excellent choice for Gomoku because it separates the estimation of state values (V(s)) and action advantages (A(s, a)). This helps the agent prioritize actions that improve the overall state value, even when the immediate rewards are sparse.


### 4. Stable Training Practices
- Experience Replay
  - Large replay buffer (e.g. 50 k transitions).
  - Sample mini‑batches uniformly or via Prioritized Experience Replay.
- Target Network
  - Clone main Q‑network every N steps (e.g. N=1 000) to compute stable targets.
- Double DQN
  - Use action argmax from main network but value from target network to reduce overestimation.
- Dueling Architecture
  - Separates state value & advantage for faster learning of where states are good vs. specific actions.
- Gradient Clipping & Learning Rate Scheduling
  - Clip gradients at e.g. |g|≤10.
  - Use Adam optimizer with lr≈1e‑4 and reduce on plateau.
- Batch Normalization &/or LayerNorm (optional) to stabilize activations.

> **_NOTE_**
> - **Experience Replay** stores past experiences (state, action, reward, next state, done flag) in a replay buffer. During training, mini-batches of transitions are sampled from this buffer instead of using only the most recent experience. This reduces correlation between consecutive experiences and reusing past experiences allows the agent to learn more from fewer interactions with the environment.
> - The **Target network** is a copy of the main Q-network that is periodically updated (e.g., every 1,000 steps) to compute stable target Q-values during training.
> - **Double DQN** addresses the issue of overestimation in Q-learning by decoupling the selection of actions from the evaluation of their values. In vanilla DQN, the same network is used to both select and evaluate actions, which can lead to overestimated Q-values and suboptimal policies. Double DQN reduces this bias by:
>   - Using the **main network** to select the best action (arg max<sub>a</sub>> Q<sub>main</sub>(s', a))
>   - Using the **target network** to evaluate the value of that action (Q<sub>target</sub>(s', a))
> - **Dueling DQN** splits the Q-value into two components, which helps learn faster:
>   - State Value (V(s)) : Represents how good a state is, regardless of the action taken.
>   - Advantage (A(s,a)) : Represents how much better a specific action is compared to the average.
> - Gradient Clipping: Limits the magnitude of gradients during backpropagation (e.g., clipping at ∣g∣≤10) to prevent exploding gradients
> - Learning Rate Scheduling: Starting with a higher learning rate allows the model to explore. Reducing the learning rate later in training helps the model to converge to a more optimal solution.


### 5. Exploration vs. Exploitation (trade-off)
- ε‑Greedy: Start ε=1.0 → decay to ε_min≈0.05 over e.g. 50 k steps.
- Noisy Nets (alternative): Replace FC layers with parameterized noise to drive exploration without ε.

> **_NOTE_**
> - In reinforcement learning:
>   - Exploration : The agent tries new actions to discover better strategies or learn about the environment.
>   - Exploitation : The agent uses its current knowledge to select the best-known action to maximize rewards.
> - Too much exploration can slow down learning by wasting time on suboptimal actions. Too much exploitation can cause the agent to get stuck in a suboptimal policy, missing out on better strategies.
> - ε-Greedy is a simple yet effective method for balancing exploration and exploitation.
>   - With probability ϵ, the agent selects a random action (exploration).
>   - With probability 1−ϵ, the agent selects the action with the highest Q-value (exploitation).
> - Start with high exploration (ϵ=1.0) and decay ϵ over time.
> - Noisy Nets replace standard fully connected (FC) layers in the neural network with layers that include parameterized noise. The noise introduces stochasticity into the network's weights, which naturally drives exploration without requiring an explicit exploration parameter like ϵ.


### 6. Parallel Self‑Play
Actors Learners
- Spawn **multiple “worker”** processes to play games in parallel, writing transitions to a shared buffer.
- A **single learner** process pulls batches, updates weights, and syncs back to workers.
- Use Ape-X / IMPALA–style architecture for throughput.

On‐Policy vs. Off‐Policy
- DQN is off‑policy, meaning it can learn from data generated by older policies (NOT the current). This makes asynchronous actors straightforward to implement

> **_NOTE_**
> - **Self-play** involves training an agent by having it play games against itself or copies of itself. This allows the agent to improve by learning from its own experiences. Parallel self-play scales this process by running multiple games simultaneously across multiple processes or machines.
> - Actors (Workers): Each worker generates transitions (state, action, reward, next state) by interacting with the environment (the Gomoku board). And write transitions to a shared replay buffer that stores all experiences for training.
> - Workers can generate transitions independently, and the learner can update the network using any batch of data from the buffer. This **decoupling of data generation and learning** makes DQN well-suited for distributed RL architectures like Ape-X and IMPALA.


### 7. Invalid‐Move Masking
Action Masking at inference and training:
- [Inference step] Before selecting argmax or sampling, set Q(a_i)=−1e9 (any very low value) for any occupied cell i.
- [Training step] Ensures no gradient signal encourages picking illegal moves.

> **_NOTE_**
> - Invalid-Move Masking ensures that the agent only selects valid moves by explicitly penalizing invalid actions (e.g., move to a cell already occupied by a stone) during both inference (action selection) and training (Q-value updates).
> - At Training (Gradient Updates) : Set the Q-values of invalid actions to −1e9 before computing the loss or gradients.


### 8. GPU Utilization
- Move model and batch tensors to GPU (e.g. device=torch.device("cuda")).
- Pin memory in DataLoader if using large replay batches.
- Profile and adjust batch size (e.g. 64–256) to maximize utilization.

> **_NOTE_**
> - Moving the model and data to the GPU ensures that all computations (forward passes, backpropagation, etc.) are performed on the GPU, which is significantly faster than using the CPU.
> - Pinned memory refers to memory allocated on the CPU that is page-locked (non-pagable). This allows for faster data transfer between the CPU and GPU.
> - The batch size determines how many transitions (state, action, reward, next state) are processed together during each training step. Small batch sizes : May underutilize the GPU, slower training but require less GPU memory.


### 9. Evaluation & Baselines
Baseline Opponents
- Random player, “Greedy” threat‐maximizer (picks move that maximizes immediate 2/3 threat count).
- Rule‐based 1‐2‐3 step lookahead.

Metrics
- Win rate over 1 000 games vs. each baseline.
- Learning curve: moving‑average win rate vs. training steps.

Checkpointing
- Save model every M steps, evaluate offline to track performance drift.

> **_NOTE_**
> - Random Player : This opponent selects moves randomly from the set of valid moves. Ensure the RL agent can beat at least the simplest baseline.
> - "Greedy" Threat-Maximizer : This opponent evaluates each move based on how much it increases its own "threats" (e.g., creating open-ended lines of 2 or 3 stones). Still short-sighted.
> - Rule-Based Lookahead (1–3 Steps) : This opponent uses a rule-based approach with limited lookahead (1, 2, or 3 steps ahead).
> - Why Use Baseline Opponents? They provide a clear progression of difficulty and assess whether the agent is learning meaningful strategies.
> - The RL agent plays 1,000 games against each baseline opponent and calculate win rate
> - Plots the moving-average win rate (e.g., over the last 100 games) against the number of training steps.
> - Checkpointing involves saving snapshots of the RL agent's model at regular intervals during training. Resume training from a specific point if needed.


### Workflow
1. Implement Gym Env with action masking & shaping.
2. Build Dueling Double-DQN with CNN backbone in PyTorch.
3. Set up Replay + Target nets + GPU training loop.
4. Launch parallel actors for high-throughput self‑play.
5. Decay ε, monitor loss and Q‐values, adjust hyperparams.
6. Evaluate periodically vs. baselines; select best checkpoints.
7. Wrap best model in Pygame interface for human play.

With these components you’ll get a robust, sample‐efficient Gomoku agent capable of learning from self‑play and providing an interactive GUI for human challenges.


## Detailed Implementation

\$ python -m venv gomoku-env

\$ source gomoku-env/Scripts/activate

\$ pip install -r requirements.txt --upgrade

### Build Game Environment 

**gomoku_env.py**
- Compatibility with OpenAI Gym : Implements the required methods (reset, step, render) and defines action_space and observation_space.
- Reward Shaping : Includes a heuristic evaluation (_evaluate_board) to provide intermediate rewards, which helps alleviate sparse rewards during training.
- Invalid Move Handling : Penalizes invalid moves (reward = -1.0) and terminates the episode, ensuring the agent learns to avoid such actions.
- **Precomputing valid actions** avoids repeatedly scanning the board for empty cells. Incremental updates ensure accuracy while minimizing overhead.
- A pattern "-XX-" can be substring of "--XX-" and "-XX--" and be counted as 2.
- test_reward.py to verify, all of the sudden, loss curve becomes unbelievable
- (Cache result for _check_line() to avoid recalculating threats for cells that haven't changed since the last move, but Clear the cache after each move to ensure accuracy)
- Cache line result. Clear caches in a region of 3x3 around the last move.
- 4 channels with 2 additional ones for threat_maps. Observation Enrichment (Adding Strategic Features): This involves modifying the environment's observation space to include more informative channels that the agent can directly use. We'll add channels for potential immediate winning moves (creating a 4-in-a-row) for both the current player and the opponent.
- More checking (for reward shaping, threat_map), slower
- Loss curve has any meaning? Not performance related. It tracks the training loss (typically MSE loss) between the predicted Q-values and the target Q-values. A low loss means:→ the network is getting better at fitting the targets it's given during training. It only tells you how well your model is fitting the experience data (replay buffer) it has seen. BUT — it does not guarantee the agent plays well. It says nothing about: Whether the experience was good. Whether the learned policy is actually strategic.
- Reduce epsilon decay rate, plot epsilon
- Ensure that the magnitude of intermediate rewards (e.g., threat-based rewards and fork rewards) is small compared to the win reward. This ensures that the agent prioritizes winning over accumulating intermediate rewards.
- intermediate rewards: REWARD_LIVE3
  - REWARD_LIVE2: weak threat
  - REWARD_SEMI_OPEN3: a moderate threat, as it requires the opponent to block to prevent a win
  - REWARD_LIVE3: a strong threat because it can easily become a winning line
  - REWARD_FORK: A fork represents a significant strategic advantage, as it forces the opponent into a defensive position.
  - As training progresses, you may want to reduce the influence of intermediate rewards (e.g., Live 2, Semi-Open 3) to encourage the agent to focus on winning. 
- To **verify reward**, we need to print out patterns detected from boards, print the board before and after a move, print 4-channel observation, print reward components to check if they match our expectation.

### Build a Deep Q-Network (DQN) Agent

**DQN**
- Enable Dropout to help reduce overfitting, especially since your board is small (36 actions max), and the model can overfit easily.
- Now your network receives direct information about where winning moves are possible for both players, which can significantly help it learn crucial strategies like blocking opponent threats and recognizing winning opportunities.

**dqn_agent.py**
- State is represented as a 3D tensor with shape (2, board_size, board_size)
- Actions is a list of integers in the range [0, board_size * board_size - 1]. Action = placing a stone on an empty cell
- Predict Q-values for all possible actions in each state in the batch. size of q_values is (batch_size, board_size * board_size)
- `q_vals` is a subset of q_values that extracts the Q-values corresponding to the actions actually taken. size of q_vals is (batch_size, 1) - squeezed to (batch_size,)
- **Targets** are the "ideal" or "desired" Q-values that the agent should learn to predict. Targets help guide the learning process by telling the network what Q-values it should aim to predict. In DQN, the target Q-values are computed using the Bellman equation , which incorporates:
  - The immediate reward received after taking an action.
  - The discounted maximum Q-value for the next state (estimated by the target network).
- The goal of reinforcement learning is to **train the Q-network to approximate the true Q-values** for all state-action pairs. By minimizing the loss between the predicted Q-values (q_vals) and the target Q-values (targets), the Q-network learns to make better predictions over time.
- Double DQN: Using the online network to select the action for the target calculation helps reduce overestimation bias. Good.
- Valid Move Masking: Absolutely essential for board games. Preventing the agent from choosing invalid moves and masking their Q-values correctly is vital. Your implementation of setting Q-values to -1e10 for invalid moves in the act function is correct.

### Training DQN Agent

**train_dqn.py**
- DQNAgent: The reinforcement learning agent that learns to play Gomoku.
- ReplayBuffer: A buffer to store past experiences (state, action, reward, next state, done) for training. The buffer ensures that the agent learns from diverse experiences rather than just the most recent ones.
- BOARD_SIZE and WIN_LENGTH: Define the size of the board and the number of consecutive stones needed to win.
- TRAIN_START: Minimum number of transitions required in the replay buffer before training begins.
- A single DQN agent that will learn to play Gomoku. This agent alternates between playing as Player 1 and Player 2 during self-play.
- For each step:
  - The agent selects an action using the act method (ε-greedy policy).
  - The environment executes the action and returns the next state, reward, and whether the episode is done.
  - The transition `(state, action, reward, next_state, done)` is stored in the `trajectory`.
- Reward Flipping :
  - In self-play, rewards are always from the perspective of the **current player**.
  - To ensure consistency, the reward is flipped for the opponent's turn (i.e., multiplied by -1) during post-processing.
  - Each corrected transition is added to the replay buffer for future training.
- Training Condition : Training begins only after the replay buffer contains at least TRAIN_START transitions.
- Target Network Synchronization : 
  - Every TARGET_UPDATE episodes, the target network is updated to match the main Q-network.
  - This ensures stable Q-value updates during training.
- TensorBoard provides interactive visualizations for tracking metrics like loss, epsilon, and rewards. It helps you identify issues like unstable training or poor convergence more easily.
- Prioritized experience replay focuses on important transitions, improving learning efficiency. It helps the agent learn faster by revisiting challenging experiences.


**greedy_agent.py**
- The agent prioritizes moves closer to the center of the board, which is a reasonable heuristic in many board games because the center provides more opportunities to form lines in multiple directions.
- The agent ensures it only considers empty cells by skipping indices where `flat_board[idx] != 0`
- Subtracting them (obs[0] - obs[1]) gives a board where:
  - 0 = empty,
  - +1 = your stone,
  - -1 = opponent's stone.
- Better than Adding them (obs[0] + obs[1]) gives a board where:
  - 0 = empty,
  - 1 = either player's stone.
- This allows the agent to **strategically evaluate moves**:
  - Check if placing your stone (+1) creates a line of 3 for you.
  - Later, you can also add logic to block the opponent’s lines (by checking -1 patterns).

