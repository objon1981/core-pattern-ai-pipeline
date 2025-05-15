# src/rl_expander.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class RLExpander:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3, gamma=0.99, device='cpu'):
        self.device = device
        self.policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []

        # Compute discounted rewards
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Calculate policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Reset rewards and log probs
        self.rewards.clear()
        self.saved_log_probs.clear()

    def train(self, env, episodes=1000):
        """
        Train the RL agent on a custom environment.

        Args:
            env: Custom environment with reset(), step(action) methods.
            episodes: Number of training episodes.
        """
        for episode in range(episodes):
            state = env.reset()
            done = False
            ep_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.rewards.append(reward)
                state = next_state
                ep_reward += reward

            self.finish_episode()

            if episode % 100 == 0:
                print(f"Episode {episode}\tReward: {ep_reward:.2f}")

    def expand(self, compressed_input, env):
        """
        Use the trained policy to expand a compressed input.

        Args:
            compressed_input: The compressed representation (state)
            env: Custom environment to step through expansion

        Returns:
            expanded_output: The expanded representation
        """
        state = compressed_input
        done = False
        expanded = []

        while not done:
            action = self.select_action(state)
            state, reward, done, info = env.step(action)
            expanded.append(action)

        return expanded
