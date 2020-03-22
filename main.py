from Agent import Agent, ReplayMemory
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
from tqdm.notebook import tqdm

np.random.RandomState(42)

n_episode = 10000
max_horizon = 1000
gamma = 0.9 
epsilon = 1.0
epsilon_decay=0.996
epsilon_end = 0.01


actions = np.array( [k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])])
nb_actions = len(actions)

env_name = "CarRacing-v0"
env = gym.make(env_name)






scores = [] # list containing score from each episode
scores_window = deque(maxlen=50) # last 100 scores
def run():
    epsilon = 0.01
    i = 0
    agent = Agent(env)
    for i_episode in tqdm(range(n_episode), desc="Episodes"):
        score = 0
        agent.reinitialisation_episode()
        for i_step in tqdm(range(max_horizon), desc="Action steps"):
            env.render()
            action = agent.take_action(epsilon)
            epsilon = max(epsilon*epsilon_decay,epsilon_end)

            score += agent.reward
            scores_window.append(score) ## save the most recent score
            scores.append(score) ## sae the most recent score

            boo = agent.learn_from_action(action)
            if boo == True:
                break

            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
            if i_episode %100==0:
                print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
                
            if np.mean(scores_window)>=300.0:
                print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                           np.mean(scores_window)))
                torch.save(agent.estimate_network.state_dict(),f'checkpoint_{i_episode}.pth')
                break
    env.close()
    return scores

if __name__ == "__main__":
    scores =run()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)),scores)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()


def evaluate(model):
    model.eval()
    obs = env.reset()
    
    frame = deque(maxlen=num_frame_stack)
    for i in range(num_frame_stack):
        frame.appendleft(to_grey(obs))
        
    reward_total = 0
    for i in tqdm(range(100*max_horizon)):
        env.render()
        proba = np.random.uniform(0, 1)
        stack = stack_to_vector(frame)
        Q_current = model(stack)
        greedy_ind = np.argmax(Q_current.detach().numpy())
        
        if (proba > 0.9):
            action = actions[greedy_ind]
            obs_next, reward, done, info = env.step(action)
            reward_total += reward
        else :
            action_ind = np.random.randint(0, nb_actions)
            action =  actions[action_ind]
            obs_next, reward, done, info = env.step(action)
            reward_total += reward
        frame.appendleft(to_grey(obs_next))
        
        if done == True:
            print('done = true')
            return reward_total, i
        
    return reward_total, 0