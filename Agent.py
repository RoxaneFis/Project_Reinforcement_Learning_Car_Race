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




from graphviz import Digraph
import torch
from torch.autograd import Variable


# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot
from Model import Q_model


all_actions = np.array( [k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])])
nb_actions = len(all_actions)
num_frame_stack = 4
learning_rate = 0.001
batch_size = 5
memory_size = int(1e5)
UPDATE_EVERY = 4 
GAMMA = 0.99 
gamma = 0.99
TAU = 1e-3 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_deque(obs,num_frame_stack=num_frame_stack):
    d = deque(maxlen=num_frame_stack)
    for i in range(num_frame_stack):
        d.appendleft(obs)
    return d

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
    def __init__(self, env):
        self.target_network = Q_model(output_dim=nb_actions, trainable=True).to(device)
        self.estimate_network = Q_model(output_dim=nb_actions, trainable=True).to(device)
        self.optimizer = optim.Adam(self.estimate_network.parameters(),lr=learning_rate)
        self.memory =ReplayBuffer(nb_actions, memory_size,batch_size,0) 
        
        self.target_parameters = self.target_network.parameters()
        self.estimate_parameters = self.estimate_network.parameters()
        self.frame = None
        self.target_network.eval()
        self.estimate_network.train()
        self.loss = []
        self.env = env
        self.t_step = 0
        self.reward =0





    def init_frame(self,obs,num_frame_stack=num_frame_stack):
        d = deque(maxlen=num_frame_stack)
        for i in range(num_frame_stack):
            d.appendleft(to_grey(obs))
        return d
    
    def reinitialisation_episode(self):
        observation = self.env.reset()
        self.frame = self.init_frame(observation)


    def update_target_network(self):
        self.target_network.load_state_dict(self.estimate_network.state_dict())                        

    
    def take_action(self, eps):
        proba = random.random()
        with torch.no_grad():
            stack = stack_to_vector(self.frame)
            Q_current = self.estimate_network(stack)
        self.estimate_network.train()
        if (proba > eps):
                greedy_ind = np.argmax(Q_current.detach().numpy())
                action = all_actions[greedy_ind]
        else :
            action_ind = np.random.randint(0, nb_actions)
            action =  all_actions[action_ind]
        return action
        #return torch.tensor(action)



    def learn_from_action(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.reward=reward
        '''
        if reward < 0:
            reward = reward*100
        if reward > 0:
            reward = reward*10
        '''
        if(done == True):
            return done
        
        stack = stack_to_vector(self.frame)
        self.frame.appendleft(to_grey(obs_next))
        stack_next = stack_to_vector(self.frame)


        self.memory.add(stack, action, reward, stack_next, done)
        #self.memory.push({"st":stack,"at":action, "rt":reward,"st+1":stack_next})
        if len(self.memory) < batch_size:
            return
        stacks, actions, rewards, stack_nexts, dones = self.memory.sample()

        loss_function = nn.MSELoss()
        self.estimate_network.train()
        self.target_network.eval()

        actions_index = torch.Tensor([find_index(all_actions,actions[i]) for i in range(batch_size)]).long()
        predicted_targets = self.estimate_network(stacks).gather(1,actions_index.view(-1,1))
        with torch.no_grad():
            labels_next = self.target_network(stack_nexts).detach().max(1)[0].unsqueeze(1)
        labels = rewards + (gamma* labels_next*(1-dones))
        loss = loss_function(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.estimate_network,self.target_network,TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)



    
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
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
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        return len(self.memory)