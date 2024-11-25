import gym 
from dqn_tf import DeepQNetwork, Agent
import numpy as np
from gym import wrappers

# Get images
def preprocess(observations):
   return np.mean(observations[30:, :], axis=2).reshape(180, 160, 1)

# Get sense of motion by stacking frames
def stack_frames(stacked_frames, frame, buffer_size):
   if stacked_frames is None:
      stacked_frames = np.zeros((buffer_size, *frame.shape))
      for idx, _ in enumerate(stacked_frames):
         stacked_frames[idx, :] = frame
   else:
      stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
      stacked_frames[buffer_size-1, :] = frame
   
   stacked_frames = stacked_frames.reshape(1, *frame.shape[0:2], buffer_size)
   return stacked_frames


         