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

# parameters

batch_size = 128
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
tau = 0.005
lr = 1e-4
