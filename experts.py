"""
File: experts.py
Description: Defines the Deep Q-Network (DQN) Agent.
             This class is instantiated multiple times to create
             the different 'Experts' (Time, Energy, Security).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import config

# --- 1. The Brain (Neural Network) ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Input Layer: Takes the State Vector [step, p_0, p_1...]
        self.fc1 = nn.Linear(state_size, 128)
        # Hidden Layer: Processes patterns
        self.fc2 = nn.Linear(128, 128)
        # Output Layer: Outputs Q-values for each location (Action)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        # ReLU activation allows learning complex non-linear patterns
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. The Memory (Replay Buffer) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves an experience."""
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly selects experiences to train on."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (torch.stack(states).to(config.DEVICE),
                torch.stack(actions).to(config.DEVICE),
                torch.stack(rewards).to(config.DEVICE),
                torch.stack(next_states).to(config.DEVICE),
                torch.stack(dones).to(config.DEVICE))

    def __len__(self):
        return len(self.buffer)

# --- 3. The Agent (The Worker) ---
class ExpertAgent:
    def __init__(self, state_size, action_size, name="Generic_Expert"):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters from config.py
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.learning_rate = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        
        # Policy Net: The one that plays the game
        self.policy_net = DQN(state_size, action_size).to(config.DEVICE)
        # Target Net: The one we calculate future rewards against (Stability)
        self.target_net = DQN(state_size, action_size).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(config.MEMORY_CAPACITY)

    def choose_action(self, state, evaluate=False):
        """
        Epsilon-Greedy Strategy.
        If evaluate=True, we turn off randomness (Pure Exploitation).
        """
        # Exploration: Pick random action
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: Pick best action based on Neural Net
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            q_values = self.policy_net(state_t)
            return torch.argmax(q_values).item()

    def learn(self):
        """Updates the Neural Network weights based on memory."""
        if len(self.memory) < self.batch_size:
            return

        # 1. Get a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # --- FIX STARTS HERE ---
        # Ensure proper shapes: (Batch_Size, 1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        # --- FIX ENDS HERE ---

        # 2. Get Q(s,a) from Policy Net
        current_q = self.policy_net(states).gather(1, actions)

        # 3. Get Max Q(s', a') from Target Net
        with torch.no_grad():
            # Get max predicted Q value for next state
            next_q = self.target_net(next_states).max(1)[0].view(-1, 1) # Ensure shape is (Batch, 1)
            
            # Bellman Equation: Reward + Gamma * Future_Reward
            # If done, future reward is 0
            target_q = rewards + (self.gamma * next_q * (~dones))

        # 4. Calculate Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6. Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        """Copies weights from Policy to Target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())