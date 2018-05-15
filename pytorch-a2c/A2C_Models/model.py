import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space, input_dims, use_cuda):   #use_cuda is not used and for compatibility reasons (I2A needs the use_cuda parameter)
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.linear1_in_dim = get_linear_dims_after_conv([self.conv1, self.conv2, self.conv3], input_dims)

        self.linear1 = nn.Linear(self.linear1_in_dim, 512)

        #self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        #self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        #self.linear1 = nn.Linear(64 * 7 * 7, 512)

        num_outputs = action_space
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        self.apply(weights_init)

        self.conv1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.conv2.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.conv3.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.linear1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu

        self.train()

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, self.linear1_in_dim)
        x = self.linear1(x)
        x = F.relu(x)

        return self.critic_linear(x), self.actor_linear(x)


def get_linear_dims_after_conv(conv_list, input_dims):
    conv_output_dims = get_conv_output_dims(conv_list, input_dims)
    return conv_list[-1].out_channels * reduce(lambda x, y: x * y, conv_output_dims)

def get_conv_output_dims(conv_list, input_dim):
    intermediate_dims = input_dim
    for conv_layer in conv_list:
        intermediate_dims = get_single_conv_output_dims(conv_layer, intermediate_dims)
    return intermediate_dims

#this method only works for 2 input dimensions
def get_single_conv_output_dims(conv_layer, input_dims):
    assert len(input_dims)==2
    out_dim_1 = ((input_dims[0] - conv_layer.kernel_size[0] + 2 * conv_layer.padding[0]) / conv_layer.stride[0]) + 1
    out_dim_2 = ((input_dims[1] - conv_layer.kernel_size[1] + 2 * conv_layer.padding[1]) / conv_layer.stride[1]) + 1
    return math.floor(out_dim_1), math.floor(out_dim_2)


