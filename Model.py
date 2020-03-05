import torch
import torch.nn.functional as F 
import torch.nn as nn


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=3,H=96, W=96):
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
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    def __init__(self, nb_frames, output_dim, batch_size=1):
        super(Q_model,self).__init__()
        self.model = nn.Sequential(
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                Unflatten(batch_size, nb_frames, 96, 96),
                #Unflatten(1, 3, 96, 96),
                nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, stride=3), #46
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2), #42
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
                nn.MaxPool2d(kernel_size=2, stride=2), #21
                Flatten(),
                nn.Linear(288, 256),
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
                nn.Linear(256, output_dim),)

    def forward(self,x):
         return self.model(x)