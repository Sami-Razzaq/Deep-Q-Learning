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
   
def optimize_model():
   
   # check to ensure enough samples for mini-batch
   if len(memory) < batch_size:
      return
   
   # extract mini-batch from Replay Memory
   # Converts batch-arrays of transitions to transitions of batch-arrays
   transition = memory.sample(batch_size)     
   batch = Transition(*zip(*transition))
   
   # Computes mask of non final states and concatenate the batch elements
   non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                           batch.next_state)), device=device)
   
   non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])
   
   state_batch = torch.cat(batch.state)
   action_batch = torch.cat(batch.action)
   reward_batch = torch.cat(batch.reward)
   
   state_action_values = policy_net(state_batch).gather(1, action_batch)
   next_state_values = torch.zeros(batch_size, device=device)
   
   with torch.no_grad():
      next_state_values[non_final_mask] = target_net(non_final_next_state).max(1)[0]
   
   # Calculate expected q-value
   expected_state_action_values = (next_state_values * gamma + reward_batch)
   
   # Compute Huber Loss
   criterion = nn.SmoothL1Loss()
   loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
   
   # Optimize the model
   optimizer.zero_grad()
   loss.backward()
   
   # In-place clipping to 100 to prevent exploding gradient problem
   torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
   optimizer.step()
   
if torch.cuda.is_available():
   num_episodes = 600
else:
   num_episodes = 300
   
for i in range(num_episodes):
   
   # initialize environment & get it's state
   state, info = env.reset()
   
   state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
   
   for t in count():
      action = select_action(state)
      observation, reward, terminated, truncated, _ = env.step(action.item())
      
      reward = torch.tensor([reward], device=device)
      
      done = terminated or truncated
      
      if terminated:
         next_state = None
         
      else:
         next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
      
      # Store transition
      memory.push(state, action, next_state, reward)
      
      state = next_state
      
      optimize_model()
      
      target_net_state_dict = target_net.state_dict()
      policy_net_state_dict = policy_net.state_dict()
      
      for key in policy_net_state_dict:
         target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1-tau)
         
      if done:
         episode_durations.append(t + 1)
         plot_duration()
         break
         
print("Complete")
plot_duration(show_result=True)
plt.ioff()
plt.show()