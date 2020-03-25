
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
from tqdm import tqdm

from Model import Q_model
from Agent import Agent, all_actions,nb_actions, to_grey, stack_to_vector

env_name = "CarRacing-v0"
env = gym.make(env_name)

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
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--model_path', type=str, default = None)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()
    return args


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
    evaluate = args.evaluate
    batch_size= args.batch_size
    learning_rate = args.learning_rate 
    tau = args.tau

    agent = Agent(env, model_path, evaluate, num_frame_stack, batch_size,learning_rate, tau)
    if (not evaluate):
        name = 'train'
    else:
        name = 'evalutate'
    with open(f"tests/{name}_{test_name}.txt", "w+") as f: 
        for i_episode in range(n_episode):
            max_score = 0
            score = 0
            agent.reinitialisation_episode()
            for i_step in range(max_horizon):
                if(render):
                    env.render()
                action = agent.take_action(eps)
                eps = max(eps*eps_decay,eps_end)
                score += agent.reward
                if score > max_score:
                    max_score = score
                boo = agent.learn_from_action(action, gamma)
                if boo == True:
                    break
                print(f'\rEpisode {i_episode} || Max Score {max_score} || End Score {score}', end="\r", flush=True)
                if i_episode %50==0 and i_episode>0:
                    torch.save(agent.estimate_network.state_dict(),f'checkpoints/checkpoint_{i_episode}_{test_name}.pth')   
                if max_score>=900.0:
                    print(f'\rEpisode {i_episode}\with excpetional Max Score {max_score}\ End Score {score}')
                    torch.save(agent.estimate_network.state_dict(),f'checkpoints/Score_850_checkpoint_{i_episode}_{test_name}.pth')
            print(f'\rEpisode {i_episode} || Max Score {max_score} || End Score {score}')
            f.writelines(f"{str(max_score)} {str(score)}\n")
    torch.save(agent.estimate_network.state_dict(),f'checkpoint_final_{test_name}.pth')         
    env.close()


def read_score(path):
    max_scores = []
    end_scores = []
    nb_success = 0
    with open(path, "r") as f: 
            for line in f:
                s = line.split()
                max_scores.append(float(s[0]))
                if (float(s[0]) >= 900):
                    nb_success = nb_success+1
                end_scores.append(float(s[1]))
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.text(10, 750, f'#Success :{nb_success}', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    x = np.linspace(0, len(max_scores), 1000)
    plt.plot(x, 900+0*x, '-r') 
    plt.plot(np.arange(len(max_scores)),max_scores, label='Max Scores')
    plt.plot(np.arange(len(end_scores)),end_scores, label='End Scores')
    plt.ylabel('Rewards')
    plt.xlabel('Epsiode #')
    plt.legend()
    plt.show()
        
   
if __name__ == "__main__":
    args = parse_args()
    test_name = args.test_name
    evaluate = args.evaluate
    scores =train()
    if (not evaluate):
        name = 'train'
    else:
        name = 'evalutate'
    read_score(f"tests/{name}_{test_name}.txt")
