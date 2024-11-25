from collection import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import patch
import numpy as np
import seaborn as sns
from tqdm import tqdm      # Progress bar
from gymnasium import gym

from agent import BlackJackAgent

env = gym.make('Blackjack-v1', sab=True, render_mode="rgb_array")

done = False
observation, info = env.reset()

action = env.action_space.sample()

observation, reward, terminated, truncated, info = env.step(action)

