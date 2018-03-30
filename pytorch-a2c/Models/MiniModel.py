import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


def xavier_weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

class MiniModel(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(MiniModel, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, 3, stride=1) #17x17
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2) #8x8

        self.linear_input_size = 16*8*8 #never trust the output in the error message (for some reason eg. 6 batches are drawn together if size is 6144)

        self.linear1 = nn.Linear(self.linear_input_size, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)

        self.conv1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.conv2.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.linear1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu

        self.train()

    def forward(self, inputs):
        #print("Input shape: ",inputs.shape)
        #for layer in self.modules():
        #    if isinstance(layer, nn.Linear):
        #        print(layer.weight.shape)

        x = F.relu(self.conv1(inputs / 255.0))

        x = F.relu(self.conv2(x))

        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.linear1(x))

        return self.critic_linear(x), self.actor_linear(x)
