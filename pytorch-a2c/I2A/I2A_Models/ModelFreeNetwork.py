import torch
import torch.nn as nn
import torch.nn.functional as F

# see B.1: model free path uses identical network as the standard model-free baseline agent (withput the fc layers)
class ModelFreeNetworkMiniPacman(nn.Module):
    def __init__(self, input_channels=1):
        super(ModelFreeNetworkMiniPacman, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)      #size-preserving
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)                  #19 input -> 10 output (for minipacman)

        #TODO (20.04): linear layer
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x