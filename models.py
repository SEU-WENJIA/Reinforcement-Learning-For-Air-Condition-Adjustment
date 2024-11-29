import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import random
import datetime
import argparse
import os
import sys
import gym



# TD3 类
class TD3():
    def __init__(self, state_dim, action_dim, max_action, device='cuda'):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = ReplayBufferTD3(1000000)
        self.device = device
        self.writer = SummaryWriter('runs/TD3')

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, batch_size, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        state, action, next_state, reward, done = self.memory.sample(batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            noise = torch.randn_like(action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * gamma * target_q.squeeze(-1)  # 确保 target_q 是一维张量

        current_q1 = self.critic_1(state, action).squeeze(-1)  # 确保 current_q1 是一维张量
        current_q2 = self.critic_2(state, action).squeeze(-1)  # 确保 current_q2 是一维张量

        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        if self.memory.size() % policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



class ReplayBufferTD3:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        state, action, next_state, reward, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.stack(action), np.stack(next_state), np.stack(reward), np.stack(done)

    def size(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    









# 局部价值函数网络
class LocalValueNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(LocalValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# 定义局部动作映射
def map_action(action):
    status = (action[0] + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]
    set_temp = abs(action[1] + 1)/2 * (28-18) + 18  # 将 [-1, 1] 映射到 [18, 26]
    return np.array([status, np.clip(set_temp, 18, 28)])

# VDN类
class VDN:
    def __init__(self, obs_dim, action_dim, device='cuda'):
        self.local_value_net = LocalValueNet(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.local_value_net.parameters())
        self.memory = ReplayBufferVDN(1000000)
        self.device = device

    def select_action(self, obs):
        local_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = torch.randn(1, 2).to(self.device)  # 随机选择动作
        return map_action(action.cpu().numpy()[0])

    def update(self, batch_size, gamma=0.99):
        if self.memory.size() < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 计算局部价值函数
        local_value = self.local_value_net(states, actions)
        next_local_value = self.local_value_net(next_states, actions)

        target = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * next_local_value  # 确保形状一致
        loss = F.mse_loss(local_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 经验回放池
class ReplayBufferVDN:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
