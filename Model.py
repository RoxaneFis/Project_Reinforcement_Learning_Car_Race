import torch
import torch.nn.functional as F 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import itertools as it

actions = np.array( [k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])])
nb_actions = len(actions)

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=4,H=96, W=96):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C,self.H, self.W)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Q_model(nn.Module):
    def __init__(self, output_dim=nb_actions, trainable=True):
        super(Q_model,self).__init__()
        def size_out(in_size, kernel_size, stride):
            return (in_size - (kernel_size - 1) - 1) // stride  + 1
        
        self.model = nn.Sequential(
                Unflatten(),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=7, stride=3), #46
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1), #42
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2), #21
                Flatten(),
                nn.Linear(16*size_out(size_out(size_out(size_out(96,7,3),2,2),3,1),2,2)**2, 256),
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
                nn.Linear(256, output_dim))
    
    def forward(self,x):
         return self.model(x)