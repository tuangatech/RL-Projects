import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from game_config import Config


class DQN(nn.Module):
    # This is a Convolutional Neural Network (CNN) for DQN
    # defines the neural network architecture used to approximate the Q-values for the DQN agent
    def __init__(self, board_size):
        super(DQN, self).__init__()
        self.board_size = board_size
        # Input channels should be 4 (2 raw board + 2 threat maps)
        input_channels = 4

        # Convolutional layers with batch normalization to extract spatial features
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)  # Input: [B, 4, 6, 6]
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers with dropout
        flattened_size = 64 * board_size * board_size
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.3)  # Dropout layer to prevent overfitting
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, board_size * board_size)

        self._init_weights()  # Initialize weights using Kaiming normal initialization

    def forward(self, x):
        # Convolutional layers with batch normalization
        x = self.activation(self.bn1(self.conv1(x))) # Input: [B, 2, 6, 6] -> Output: [B, 32, 6, 6]
        x = self.activation(self.bn2(self.conv2(x))) # Output: [B, 64, 6, 6]

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1) # Flatten the tensor to [B, 64 * board_size * board_size]

        # Fully connected layers with dropout
        x = self.activation(self.fc1(x)) # Output: [B, 128]
        x = self.dropout(x)      # Apply dropout
        x = self.activation(self.fc2(x))
        return self.out(x)  # Output: [B, board_size * board_size]

    # Initialize weights using Kaiming normal initialization
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DQNAgent:
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
        self.update_target()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), weight_decay=1e-4, lr=Config.LEARNING_RATE)
        # Track frames for learning rate scheduling
        self.frame_count = 0    # a frame is an interaction with the environment (transition)
        
        # Lambda function for learning rate decay
        # This creates a linear decay from initial LR to minimum LR over specified frames
        lr_lambda = lambda frame: max(
            Config.MIN_LEARNING_RATE / Config.LEARNING_RATE,  # Minimum LR ratio
            1.0 - max(0, frame - Config.LR_DECAY_START_FRAME) / 
                 (Config.TOTAL_FRAMES - Config.LR_DECAY_START_FRAME)
        )        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.loss_fn = nn.MSELoss()

        self.gamma = Config.GAMMA                   # Discount factor for future rewards
        self.epsilon = Config.EPSILON_START         # Initial exploration rate
        self.epsilon_min = Config.EPSILON_MIN       # Minimum exploration rate
        self.epsilon_decay = Config.EPSILON_DECAY   # Decay rate for exploration 0.995

    # Update the target network with the weights of the Q-network
    # This is done periodically to stabilize training
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # Selects an action based on the current state
    # Epsilon-greedy strategy: with probability epsilon, select a random action (exploration)
    # When playing against humans, epsilon is set to 0.0 to always select the best action (exploitation)
    # When training, epsilon is decayed over time to reduce exploration and increase exploitation
    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        # Selects a random valid action (exploration)
        if np.random.rand() < epsilon:
            # Masked random move
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
        q_values[mask == 0] = -1e10  # mask == 0 identifies the indices of occupied cells

        return torch.argmax(q_values).item()

    # Update the learning rate scheduler
    # This is called every time a batch is processed
    def update_lr_scheduler(self):
        """Update learning rate based on frames processed"""
        self.frame_count += 1
        self.scheduler.step()

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

        # DQN target: use main net for argmax, target net for Q-value
        # mitigates overestimation bias by decoupling action selection and evaluation
        with torch.no_grad():
            # Use q_net to select best next actions (Double DQN trick)
            next_q_values = self.q_net(next_states)
            next_actions = torch.argmax(next_q_values, dim=1)

            # Use target_net to evaluate the Q-values of the selected actions
            next_target_q = self.target_net(next_states)
            target_q = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # max_next_q_vals = next_q_values.max(dim=1)[0]
            # Compute the target Q-values using the Bellman equation
            # targets = rewards + self.gamma * max_next_q_vals * (~dones)
            targets = rewards + self.gamma * target_q * (~dones)
            targets = torch.clamp(targets, -2, 2)     # gradient clipping to prevent exploding gradients (targets, -10, 10)

        # Computes the loss between the predicted Q-values and the target Q-values.
        loss = self.loss_fn(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Update learning rate at the end of each training step
        self.update_lr_scheduler()

        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
