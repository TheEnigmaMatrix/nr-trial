import random
from collections import deque
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from rl_sdn_controller.ai.models import DQN, DuelingDQN


class ReplayBuffer:
    """Experience Replay Buffer for Q-Learning transition storage."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for SDN Routing Control.
    Implements epsilon-greedy exploration, experience replay optimization,
    and target network updates.
    """
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.lr = config.get("learning_rate", 0.0005)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 10)
        hidden_dims = config.get("network_hidden", [128, 128])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize online and target Q-networks
        use_dueling = config.get("use_dueling", True)
        if use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim, hidden_dims).to(self.device)
        else:
            self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=config.get("buffer_capacity", 10000))
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()
        return torch.argmax(q_values, dim=1).item()

    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Compute target Q(s', a') using Double Q-Learning (DDQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        # Smooth L1 / Huber Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()
