import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from game_config import Config


class DQN(nn.Module):
    # This is a Convolutional Neural Network (CNN) for DQN
    # defines the neural network architecture used to approximate the Q-values for the DQN agent
    # Dueling DQN Architecture 
    def __init__(self, board_size):
        super(DQN, self).__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size # Store action size for convenience

        # Input channels should be 2 raw boards, NO threat maps
        input_channels = 2

        # Convolutional layers with batch normalization to extract spatial features
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)  # Input: [B, 4, 6, 6]
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)   # Output: [B, 64, 6, 6]
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Add third conv
        # self.bn3 = nn.BatchNorm2d(128)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers
        flattened_size = 64 * board_size * board_size
        # --- Shared Fully Connected Layer (Optional but common) ---
        fc_output_size = 128 # Size of the output from the shared layer
        self.fc_shared = nn.Linear(flattened_size, fc_output_size) # FC layer: 64 * 6*6 = 2304 → 128
        # self.dropout = nn.Dropout(0.3) # Apply after shared FC

        # --- Value Stream ---
        # Takes the output of the shared layer and outputs a single value (for the state)
        # You can use a new layer or repurpose fc2, but let's create new ones for clarity
        self.fc_value_stream = nn.Linear(fc_output_size, 1) # Outputs state value [B, 1]

        # --- Advantage Stream ---
        # Takes the output of the shared layer and outputs an advantage for each action
        # You can use a new layer or repurpose fc2/out, but let's create new ones
        self.fc_advantage_stream = nn.Linear(fc_output_size, self.action_size) # Outputs advantages [B, action_size]

        self._init_weights()  # Initialize weights using Kaiming normal initialization

    def forward(self, x):
        # Convolutional layers with batch normalization
        x = self.activation(self.bn1(self.conv1(x))) # Input: [B, 2, 6, 6] -> Output: [B, 32, 6, 6]
        x = self.activation(self.bn2(self.conv2(x))) # Output: [B, 64, 6, 6]
        # x = self.activation(self.bn3(self.conv3(x))) # Output: [B, 128, 6, 6]

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1) # Flatten the tensor to [B, 64 * board_size * board_size]

        # --- Shared Fully Connected Layer ---
        x = self.activation(self.fc_shared(x))   
        # x = self.dropout(x) # Apply dropout after the shared layer

        # --- Split into Value and Advantage Streams ---

        # Value Stream
        # Pass through the value stream layers. No activation on the final output.
        value = self.fc_value_stream(x) # Output shape: [B, 1]

        # Advantage Stream
        # Pass through the advantage stream layers. No activation on the final output.
        advantage = self.fc_advantage_stream(x)  # Output shape: [B, action_size]

        # --- Combine Value and Advantage to get Q-values ---

        # Calculate the mean of the advantages across the action dimension for each state in the batch.
        # The result will have shape [B, 1]
        mean_advantage = advantage.mean(dim=1, keepdim=True)

        # Combine using the Dueling formula: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')) )
        # value shape: [B, 1], advantage shape: [B, action_size], mean_advantage shape: [B, 1]
        # Broadcasting in PyTorch handles the addition correctly. V and mean_advantage
        # will be broadcast to the shape of advantage for the operations.
        # q_values = torch.tanh(value + advantage - mean_advantage)
        q_values = value + advantage - mean_advantage

        # The output shape is [B, action_size], which is exactly what DQNAgent expects.
        return q_values

    # Initialize weights using Kaiming normal initialization
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DQNAgent:
    toggle1 = True  # For debugging purposes, to print the min/max of targets and Q-values
    toggle2 = True  # For debugging purposes, to print normalized gradients
    def __init__(self, board_size, device="cuda"):
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize 2 identical neural networks: the Q-network and target network
        # The Q-network is used to select actions
        # The target network is used to compute target Q-values
        self.q_net = DQN(board_size).to(self.device)
        self.target_net = DQN(board_size).to(self.device)
        self.soft_update_target()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), weight_decay=1e-4, lr=Config.LEARNING_RATE)
        # Track frames for learning rate scheduling
        self.frame_count = 0    # a frame is an interaction with the environment (transition)
        
        # --- Learning Rate Decay using LinearLR ---
        # We want to decay the learning rate linearly from Config.LEARNING_RATE
        # to Config.MIN_LEARNING_RATE over a specified number of steps.
        # Calculate the number of steps over which decay occurs
        decay_steps = max(0, Config.TOTAL_FRAMES - Config.LR_DECAY_START_FRAME)

        # Calculate the end learning rate ratio relative to the initial LR
        end_lr_ratio = Config.MIN_LEARNING_RATE / Config.LEARNING_RATE

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0, # Start at the initial LR
            end_factor=end_lr_ratio, # Decay to the minimum LR ratio
            total_iters=decay_steps # Number of steps over which decay happens
        )

        self.loss_fn = nn.MSELoss()

        self.gamma = Config.GAMMA                   # Discount factor for future rewards
        self.epsilon = Config.EPSILON_START         # Initial exploration rate
        self.epsilon_min = Config.EPSILON_MIN       # Minimum exploration rate
        self.epsilon_decay = Config.EPSILON_DECAY   # Decay rate for exploration 0.995

    # Selects an action based on the current state
    # Epsilon-greedy strategy: with probability epsilon, select a random action (exploration)
    # When playing against humans, epsilon is set to 0.0 to always select the best action (exploitation)
    # When training, epsilon is decayed over time to reduce exploration and increase exploitation
    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        # Selects a random valid action (exploration)
        if np.random.rand() < epsilon:
            # state[0], state[1] are binary matrices of size 6x6 indicating where stones are located
            # Flatten the state (a vector of size 36) and find valid actions (where the sum of both matrices is 0)
            # This means that the cell is empty and can be played
            flat_state = (state[0] + state[1]).reshape(-1)
            valid_actions = [i for i, v in enumerate(flat_state) if v == 0]
            return random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state_tensor)
        q_values = q_values.squeeze()

        # Mask invalid moves
        # The mask is a tensor of size 36, where 1 indicates a valid move and 0 indicates an invalid move
        # The mask is created by checking where the sum of both matrices is 0 (indicating an empty cell)
        mask = torch.FloatTensor((state[0] + state[1]).reshape(-1) == 0).to(self.device)
        q_values[mask == 0] = -100  # mask == 0 identifies the indices of occupied cells

        # Select the action with the highest Q-value among valid actions
        action = torch.argmax(q_values).item()

        return action


    # Train step: sample a batch from the replay buffer and train the agent
    # The batch contains (state, action, reward, next_state, done) tuples
    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch

        # state is represented as a 3D tensor with shape (2, board_size, board_size)
        # neural network expects input tensor of type FloatTensor (not LongTensor) to perform gradients and backpropagation
        states = torch.FloatTensor(states).to(self.device)
        # actions is a list of integers in the range [0, board_size * board_size - 1]
        # action = placing a stone on an empty cell
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # predict Q-values for all possible actions in each state in the batch
        # size of q_values is (batch_size, board_size * board_size)
        q_values = self.q_net(states)
        # q_vals is a subset of q_values that extracts the Q-values corresponding to the actions actually taken
        # size of q_vals is (batch_size, 1) - squeezed to (batch_size,)
        q_vals = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            # Double DQN trick to mitigate the overestimation bias of vanilla DQN 
            # by decoupling the action selection from the value evaluation
            # 1. Use the online network (`self.q_net`) to select the best action in the next state
            next_q_values = self.q_net(next_states)
            next_actions = torch.argmax(next_q_values, dim=1)

            # 2. Use the target network (`self.target_net`) to evaluate the Q-value of the action selected in step 1
            next_target_q = self.target_net(next_states)
            target_q = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # max_next_q_vals = next_q_values.max(dim=1)[0]
            # Compute the target Q-values using the Bellman equation
            # with self.gamma * target_q * (~dones) is the discounted future reward for the next state
            # `targets` represents the TD (Temporal Difference) target, which is the 
            # best estimate of the total return for the next state
            targets = rewards + self.gamma * target_q * (~dones)
            # limit the range of target values to prevent extreme updates to the network weights
            
            if self.toggle1 and 26000 <= self.frame_count:
                print(f"> Frame {self.frame_count}:")
                print(f"   Rewards: min={rewards.min().item():.3f}, max={rewards.max().item():.3f}")
                print(f"   Q-values: min={q_vals.min().item():.3f}, max={q_vals.max().item():.3f}")
                print(f"   Target Q: min={target_q.min().item():.3f}, max={target_q.max().item():.3f}")
                print(f"   Targets: min={targets.min().item():.3f}, max={targets.max().item():.3f}")
                self.toggle1 = False
            targets = torch.clamp(targets, -1, 3)     # (targets, -10, 10)

        # Computes the loss between the predicted Q-values and the target Q-values.
        loss = self.loss_fn(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping to prevent exploding gradients during backpropagation
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 3)
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        if self.toggle2 and self.frame_count > 26000:
            print(f"> Gradient norm: {grad_norm.item()}")
            self.toggle2 = False
        
        self.optimizer.step()

        # Update learning rate at the end of each training step --> move to training loop to update lr every environment step
        # self.update_lr_scheduler()
        # Update the target network using soft update after every training step 
        # (not like hard update every 200 episodes)
        self.soft_update_target()

        return loss.item()

    # Update the target network with the weights of the Q-network
    # This is done periodically to stabilize training
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # The target network is updated incrementally using a weighted average of its current weights 
    # and the Q-network's weights. This provides smoother convergence and reduces oscillations during training
    def soft_update_target(self):
        """
        Perform a soft update of the target network's parameters.
        """
        tau = Config.UPDATE_TAU  # Soft update coefficient (0 < tau << 1)
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

    # Update the learning rate scheduler
    def update_learning_params(self):
        """Update learning rate and beta (if PER) based on frames processed."""
        # self.frame_count is incremented in the main loop BEFORE this call.

        # Step the scheduler only after the decay start frame is reached
        if self.frame_count > Config.LR_DECAY_START_FRAME:
            # LinearLR steps based on how many times .step() is called
            # It will decay over 'total_iters' calls to .step()
            self.scheduler.step()

    # agent shifts from exploring the environment to exploiting over episodes (rather than frames/transitions)
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
