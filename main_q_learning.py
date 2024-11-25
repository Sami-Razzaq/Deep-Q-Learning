from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm      # Progress bar
import gymnasium as gym
from gym.wrappers import RecordEpisodeStatistics

from pdb import set_trace

env = gym.make('Blackjack-v1', sab=True, render_mode="rgb_array")

np.bool8 = np.bool_

learning_rate = 0.01
n_episodes = 30000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes/2)     # reduce exploration over time
final_epsilon = 0.1

class BlackJackAgent:
   def __init__(self, learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95):
      """
      Initialize RL Agent with empty dict of state-action value (q_values), learning rate and epsilon
      discount_factor = computes Q-Values
      """
      self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
      self.lr = learning_rate
      self.discount_factor = discount_factor
      self.epsilon = initial_epsilon
      self.epsilon_decay = epsilon_decay
      self.final_epsilon = final_epsilon
      
      self.training_error = []
   
   def get_action(self, obs: tuple[int, int, bool]) -> int:
      """
      Returns best action with prob. 1-epsilon otherwise random
      action with prob epsilon to ensure exploration
      """
      if np.random.random() < self.epsilon:
         return env.action_space.sample()
      else:
         return int(np.argmax(self.q_values[obs]))
   
   def update(self, obs: tuple[int, int, bool], action: int, 
              reward: float, terminated: bool, 
              next_obs: tuple[int, int, bool]):
      """
      Updates Q-Value of actions
      """
      future_q_value = (not terminated) * np.max(self.q_values[next_obs])
      
      temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
      
      self.q_values[obs][action] = self.q_values[obs][action] + self.lr*temporal_difference
      
      self.training_error.append(temporal_difference)
      
   def decay_epsilon(self):
      self.epsilon = max(self.final_epsilon, (self.epsilon-self.epsilon_decay))

agent = BlackJackAgent(
   learning_rate=learning_rate,
   initial_epsilon=start_epsilon,
   epsilon_decay=epsilon_decay,
   final_epsilon=final_epsilon
)

# Create environment to get the game statistics as agent trains
env = RecordEpisodeStatistics(env, deque_size=n_episodes)

# start training
for episode in tqdm(range(n_episodes)):
   obs, info = env.reset()
   done = False
   
   # Do Action, review rewards and update next action/epsilon value
   while not done:
      action = agent.get_action(obs)
      next_obs, reward, terminated, truncated, info = env.step(action)
      
      agent.update(obs, action, reward, terminated, next_obs)
      frame = env.render()
      
      
      # Show current game frame
      # plt.imshow(frame)
      # plt.show()
      # set_trace()
      
      done = terminated or truncated
      obs = next_obs
   
   agent.decay_epsilon()
   
roling_length = 500
fig, ax = plt.subplots(ncols=3)

ax[0].set_title("Episode Rewards")
reward_moving_average = (np.convolve(np.array(env.return_queue).flatten(), np.ones(roling_length), mode='valid')/roling_length)
ax[0].plot(range(len(reward_moving_average)), reward_moving_average)

ax[1].set_title("Episode Lengths")
length_moving_average = (np.convolve(np.array(env.length_queue).flatten(), np.ones(roling_length), mode='same')/roling_length)
ax[1].plot(range(len(length_moving_average)), length_moving_average)

ax[2].set_title("Training Error")
training_error_moving_average = (np.convolve(np.array(agent.training_error).flatten(), np.ones(roling_length), mode='same')/roling_length)
ax[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

plt.show()
fig.savefig(fname="Simple Q-Learning Metrics.jpg")