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


from Model import Q_model


def stack_to_vector(deque : deque):
    array = np.array(deque)
    return torch.from_numpy(array).float()
    
def to_grey(obs):
    return color.rgb2gray(obs) 

def find_index(all_actions, action):
    for i in range(len(all_actions)):
        if (np.array_equal(all_actions[i],action)):
            return i


class Agent():
    def __init__(self, env, memory, batch_size, gamma, epsilon,learning_rate, all_actions,num_frame_stack):
        self.env=env
        self.memory = memory
        self.batch_size=batch_size
        self.gamma=gamma
        self.epsilon=epsilon
        self.all_actions= all_actions
        self.nb_actions = len(all_actions)
        self.num_frame_stack = num_frame_stack 

        self.target_network = Q_model(nb_frames=num_frame_stack, output_dim=self.nb_actions)
        self.estimate_network = Q_model(nb_frames=num_frame_stack, output_dim=self.nb_actions)
        self.optimizer = optim.Adam(self.estimate_network.parameters(),lr=learning_rate)
        self.target_parameters = self.target_network.parameters()
        self.estimate_parameters = self.estimate_network.parameters()
        self.frame = None

        self.target_network.eval()
        self.estimate_network.train()

    def init_frame(self, obs):
        d = deque(maxlen=self.num_frame_stack)
        for i in range(self.num_frame_stack):
            d.appendleft(to_grey(obs))
        return d
    
    def reinitialisation_episode(self):
        observation = self.env.reset()
        self.frame = self.init_frame(observation)


    def update_target_network(self, tau = 0.1):
        for target_param, local_param in zip(self.target_network.parameters(),self.estimate_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
               

    def take_action(self):
        proba = np.random.uniform(0, 1)
        if (proba > self.epsilon) :
            with torch.no_grad():
                stack = stack_to_vector(self.frame)
                Q_current = self.estimate_network((stack))
                #select the best action
                greedy_ind = np.argmax(Q_current.detach().numpy())
                action = self.all_actions[greedy_ind]
        else :
            action_ind = np.random.randint(0, self.nb_actions)
            action =  self.all_actions[action_ind]
        return action

    def learn_from_action(self, action):
        obs_next, reward, done, info = self.env.step(action)
        stack = stack_to_vector(self.frame)
        self.frame.appendleft(to_grey(obs_next))
        stack_next = stack_to_vector(self.frame)
        #store trnasition
        self.memory.append({"st":stack,"at":action, "rt":reward,"st+1":stack_next})
        #sample minibatch
        selection = numpy.random.choice(self.memory, size=self.batch_size, replace=True)

        loss_function = nn.MSELoss()

        targets=torch.empty(self.batch_size)
        predicted=torch.empty(self.batch_size)

        for m in range(self.batch_size):
            action = selection[m]["at"]
            index_action = find_index(self.all_actions, action)
            with torch.no_grad():
                targets[m]= selection[m]["rt"]
                if (done == False):
                    stack_next = selection[m]["st+1"]
                    greedy_max = np.max(self.target_network(stack_next).detach().numpy())
                    targets[m] =+ self.gamma*greedy_max
            stack = selection[m]["st"]
            predicted[m] = self.estimate_network(stack)[0][index_action]
        
        loss = loss_function(predicted, targets)
        #tab_loss.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()





