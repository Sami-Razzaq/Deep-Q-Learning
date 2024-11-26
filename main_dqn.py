import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import deque, namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pdb import set_trace

env = gym.make('CartPole-v1')

device = torch.device("cuda" if torch.cuda.is_available() else ("cpu"))

Transition = namedtuple('Transition', 'state', 'action', 'next_state', 'reward')

class ReplayMemory(object):
   
   def __init__(self, capacity):
      self.memory = deque([], maxlen=capacity)
      
   def push(self, *args):
      """Saves a transition"""
      self.memory.append(Transition(*args))
      
   def sample(self, batch_size):
      """Samples random values into buffer"""
      return random.sample(self.memory, batch_size)
   
   def __len__(self):
      """Returns length of buffer"""
      return len(self.memory)
   
class DQN(nn.Module):
   """Multilayer Perceptron of 3-Layers
      n_observations is the environment state
      n_actions is the actions in response to the
      environment state
   """
   
   def __init__(self, n_observations, n_actions):
      super(DQN, self).__init__()
      
      self.layer1 = nn.Linear(n_observations, 128)
      self.layer2 = nn.Linear(128, 128)
      self.layer3 = nn.Linear(128, n_actions)
      
   def forward(self, x):
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      return self.layer3(x)

# hyper-parameters

batch_size = 128
gamma = 0.99         # discount factor
eps_start = 0.9      # Epsilon Starting Value
eps_end = 0.05       # Epsilon Ending Value
eps_decay = 1000     # Epsilon Decay (higher is slower)
tau = 0.005          # update rate of target network
lr = 1e-4

# Actions & States of environment 
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)         # No. of features in state
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

# Stores agent's experiences for training
memory = ReplayMemory(10000)

# No. of steps taken by agent
steps_done = 0

def select_action(state):
   """input - Current State
      returns - actions with highest value or random
   """
   global steps_done
   sample = random.random()
   eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)

   steps_done += 1
   if sample > eps_threshold:
      with torch.no_grad():
         return policy_net(state).max(1)[1].view(1,1)
   else:
      return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# durat
episode_durations = []

def plot_duration(show_result=False):
   plt.figure(1)
   duration_t = torch.tensor(episode_durations, dtype=torch.float)
   
   if show_result:
      plt.title("Result")
   else:
      plt.clf()
      plt.title("Training")
   plt.xlabel("Episode")
   plt.ylabel("Duration")
   plt.plot(duration_t.numpy())
   
   if len(duration_t) >= 100:
      means = duration_t.unfold(0, 1000, 1).mean(1).view(-1)
      means = torch.cat((torch.zero(99), means))
      
      plt.plot(means.numpy)
   
   plt.pause(0.001)