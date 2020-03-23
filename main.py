
import argparse
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

from Model import Q_model
from Agent import Agent, ReplayMemory, all_actions,nb_actions, to_grey, stack_to_vector

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--test_name', type=str, default='default1')
    parser.add_argument('--n_episode', type=int, help='nombre dépisodes', default=300)
    parser.add_argument('--max_horizon', type=int,default=1000)
    parser.add_argument('--num_frame_stack', type=int,default=4)
    parser.add_argument('--gamma', type=float, help='gamma', default=0.9)
    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--eps_decay', type=float, default=0.996)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--model_path', type=str, default = None)
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()
    return args

np.random.RandomState(42)
scores = [] # list containing score from each episode


env_name = "CarRacing-v0"
env = gym.make(env_name)
model_path ="/Users/roxanefischer/Documents/cours/3A/Advanced_Topics_in_Artificial_Intelligence/projet/code/checkpoint_288.pth"


def train():
    #HYPERPARAMETERS
    args = parse_args()
    test_name = args.test_name
    n_episode = args.n_episode
    max_horizon = args.max_horizon
    num_frame_stack = args.num_frame_stack
    gamma = args.gamma
    eps = args.eps
    eps_decay = args.eps_decay
    eps_end= args.eps_end
    model_path = args.model_path
    render = args.render

    i = 0
    agent = Agent(env, model_path)
    with open(f"test_{test_name}.txt", "w+") as f: 
        for i_episode in tqdm(range(n_episode), desc="Episodes"):
            scores_window = deque(maxlen=50) 
            score = 0
            agent.reinitialisation_episode()
            for i_step in tqdm(range(max_horizon), desc="Action steps"):
                if(render):
                    env.render()
                action = agent.take_action(eps)
                eps = max(eps*eps_decay,eps_end)
                score += agent.reward
                scores_window.append(score) ## save the most recent score
                boo = agent.learn_from_action(action)
                if boo == True:
                    break
                print(f'\rEpisode {i_episode}\tAverage Score {np.mean(scores_window)}')
                if i_episode %100==0 and i_episode>0:
                    torch.save(agent.estimate_network.state_dict(),f'checkpoints/checkpoint_{i_episode}{test_name}.pth')   
                if np.mean(scores_window)>=800.0:
                    print(f'\rEpisode {i_episode-100}\tAverage Score {np.mean(scores_window)}')
                    torch.save(agent.estimate_network.state_dict(),f'checkpoints/checkpoint_{i_episode}{test_name}.pth')
            f.writelines(f"{str(score)}\n")
            scores.append(score)
    torch.save(agent.estimate_network.state_dict(),f'checkpoint.pth')         
    env.close()
    return scores



def evaluate(path_model):
    model = Q_model()
    model.load_state_dict(torch.load(path_model))
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
            action = all_actions[greedy_ind]
            obs_next, reward, done, info = env.step(action)
            reward_total += reward
        else :
            action_ind = np.random.randint(0, nb_actions)
            action =  all_actions[action_ind]
            obs_next, reward, done, info = env.step(action)
            reward_total += reward
        frame.appendleft(to_grey(obs_next))
        if done == True:
            print('done = true')

    return (reward_total, i)



def read_score(path):
    scores = []
    with open(path, "r") as f: 
            for line in f:
                scores.append(float(line))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)),scores)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()
        

        
if __name__ == "__main__":
    args = parse_args()
    test_name = args.test_name
    scores =train()
    read_score(f"test_{test_name}.txt")
