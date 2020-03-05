from Agent import Agent
import numpy
import numpy as np
import itertools as it 
import gym
import torch
import torch.nn.functional as F 
import torch.nn as nn
from torch.autograd import Variable
from skimage import color
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

np.random.RandomState(42)

n_episode = 20
max_horizon = 100
batch_size = 10
gamma = 0.9 
num_frame_stack = 4
epsilon = 0.1
learning_rate = 0.01


all_actions = np.array( [k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])])
nb_actions = len(all_actions)

env_name = "CarRacing-v0"
env = gym.make(env_name)

def main():
    memory = list()
    agent = Agent(env, memory, batch_size, gamma, epsilon,learning_rate, all_actions, num_frame_stack)
    for i_episode in range(n_episode):
        print(f"ie episode : {i_episode}")
        agent.reinitialisation_episode()
        for i_step in range(max_horizon):
            action = agent.take_action()
            agent.learn_from_action(action)

if __name__ == "__main__":
    main()