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