import random 
from collections import deque
import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Add the tuple to the buffer
        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size: int, device: str):
        # Randomly sample a batch of transitions from the buffer
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions) 

        return (torch.tensor(states, dtype=torch.float32).to(device),
                torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(next_states, dtype=torch.float32).to(device),
                torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim, n_actions))
                                
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the policy network.
        Args:
            obs (torch.Tensor): The observation tensor, with shape [batch_size, obs_dim].
        Returns:
            torch.Tensor: The Q-values for each action.
        """
        return self.net(obs)

class DQNAgent:
    def __init__(self, obs_dim, hidden_dim, n_actions, lr, gamma, buffer_capacity, device):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        # create the Q-network
        self.q_net = QNetwork(obs_dim, hidden_dim, n_actions).to(device)
        # create the optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        # create the replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon):
        # Exploration: Random Action
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        # Exploitation: Greedy Action
        else:
            # convert state to tensor on device 
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            # get Q-values from network:
            q_values = self.q_net(state_t)

            # pick index of the max value 
            return q_values.argmax().item()

    def update(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None # not enough data
        
        # 1. Get the batch of data
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size, self.device)
        
        # 2. Compute current Q(s, a)
        current_q = self.q_net(states).gather(1, actions)

        # 3. Compute target Q(s', a')
        with torch.no_grad():
            # .max(1)[0] gives values, .max(1)[1] gives indices
            max_next_q = self.q_net(next_states).max(1)[0].unsqueeze(1)

            # Bellman equation 
            target_q = rewards + (self.gamma * max_next_q * (~dones))
        
        # 4. Compute the loss
        loss = nn.MSELoss()(current_q, target_q)

        # 5. Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

            
