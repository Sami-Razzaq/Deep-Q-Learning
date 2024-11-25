from collection import defaultdict
import numpy as np
from gymnasium import gym

from main import env

class BlackJackAgent:
   def __init__(self, learning_rate, initial_epsilon, 
                epsilon_decay, final_epsilon, discount_factor=0.95):
      """
      Initialize RL Agent with empty dict of state-action value (q_values), learning rate and epsilon
      discount_factor = computes Q-Values
      """
      self.q_values = defaultdict(lambda:np.zeros(env.action_space.n))
      self.lr = learning_rate
      self.discount_factor = discount_factor
      self.epsilon = initial_epsilon
      self.epsilon_decay = epsilon_decay
      self.final_epsilon = final_epsilon
      
      self.training_error = []
   
   def get_action(self, obs:tuple[int,int,bool]) -> int:
      """
      Returns best action with prob. 1-epsilon otherwise random
      action with prob epsilon to ensure exploration
      """
      if np.random.random() < self.epsilon:
         return env.action_sample.sample()
      else:
         return int(np.argmax(self.q_values(obs)))
   
   def update(self, obs:tuple[int,int,bool], action, reward,
              terminated, next_obs:tuple[int,int,bool]):
      """
      Updates Q-Value of actions
      """
      future_q_value = (not terminated) * np.max(self.q_values[next_obs])
      
      temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
      
      self.q_values[obs][action] = self.q_values[obs][action] + self.lr*temporal_difference
      
      self.training_error.append(temporal_difference)
      
   def decay_epsilon(self):
      self.epsilon = max(self.final_epsilon, (self.epsilon-self.epsilon_decay))