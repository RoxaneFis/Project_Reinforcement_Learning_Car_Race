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
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import random

from Model import Q_model

#SOURCE : https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda
#ReplayBuffer, soft_update are taken from this source.

all_actions = np.array( [k for k in it.product([-1, 0, 1], [1, 0], [0.3, 0])])
nb_actions = len(all_actions)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
memory_size = int(1e5)

def stack_to_vector(deque : deque):
    array = np.array(deque)
    return torch.from_numpy(array).float()
    
def to_grey(obs):
    return color.rgb2gray(obs) 

def find_index(actions, action):
    for i in range(len(actions)):
        if (np.array_equal(actions[i],action)):
            return i

class Agent():

    def load_model(self, path):
        model = Q_model(input_dim=self.num_frame_stack,output_dim=nb_actions)
        model.load_state_dict(torch.load(path))
        self.estimate_network = model
        self.target_network = model
        self.target_network.eval()
        self.estimate_network.train()

    def __init__(self, env, path=None, evaluate=False, num_frame_stack=4, batch_size=5, learning_rate=0.001, tau=1e-3):
        self.target_network = Q_model(input_dim = num_frame_stack,output_dim=nb_actions).to(device)
        self.estimate_network = Q_model(input_dim = num_frame_stack,output_dim=nb_actions).to(device)
        self.optimizer = optim.Adam(self.estimate_network.parameters(),lr=learning_rate)
        self.memory =ReplayBuffer(nb_actions, memory_size,batch_size,0) 
        self.frame = None
        self.target_network.eval()
        self.estimate_network.train()
        self.env = env
        self.reward =0
        self.evaluate = evaluate
        self.num_frame_stack = num_frame_stack
        self.batch_size=batch_size
        self.tau=tau
        if path is not None:
            self.load_model(path)


    def init_frame(self,obs):
        d = deque(maxlen=self.num_frame_stack)
        for i in range(self.num_frame_stack):
            d.appendleft(to_grey(obs))
        return d
    
    def reinitialisation_episode(self):
        observation = self.env.reset()
        self.frame = self.init_frame(observation)
               

    def take_action(self, eps):
        proba = random.random()
        with torch.no_grad():
            stack = stack_to_vector(self.frame)
            Q_current = self.estimate_network(stack)
        self.estimate_network.train()
        if (proba > eps or self.evaluate):
                greedy_ind = np.argmax(Q_current.detach().numpy())
                action = all_actions[greedy_ind]
        else :
            action_ind = np.random.randint(0, nb_actions)
            action =  all_actions[action_ind]
        return action


    def learn_from_action(self, action, gamma):
        obs_next, reward, done, info = self.env.step(action)
        self.reward=reward
        if(done == True):
            return done
        
        stack = stack_to_vector(self.frame)
        self.frame.appendleft(to_grey(obs_next))
        stack_next = stack_to_vector(self.frame)

        if(not self.evaluate):
            self.memory.add(stack, action, reward, stack_next, done)
            if len(self.memory) < self.batch_size:
                return
            stacks, actions, rewards, stack_nexts, dones = self.memory.sample()
            loss_function = nn.MSELoss()
            self.estimate_network.train()
            self.target_network.eval()
            actions_index = torch.Tensor([find_index(all_actions,actions[i]) for i in range(self.batch_size)]).long()
            estimate_rewards = self.estimate_network(stacks).gather(1,actions_index.view(-1,1))
            with torch.no_grad():
                target_expectation = self.target_network(stack_nexts).detach().max(1)[0].unsqueeze(1)
            target_rewards = rewards + (gamma* target_expectation*(1-dones))
            loss = loss_function(estimate_rewards,target_rewards).to(device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update(self.estimate_network,self.target_network,self.tau)
        return done

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


        
class ReplayBuffer:
    def __init__(self, action_size, memory_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory,k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        return len(self.memory)