import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def broadcast_action(action, num_actions, broadcast_to_shape, use_cuda):
    assert(len(broadcast_to_shape)==2)
    one_hot = torch.zeros(action.shape[0], num_actions)
    if use_cuda:
        one_hot = one_hot.cuda()
    # make one-hot vector
    one_hot.scatter_(1, action, 1)
    one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)
    # broadcast to width and height of broadcast_to_shape
    broadcasted_action = one_hot.repeat(1, 1, *broadcast_to_shape)
    return broadcasted_action

def compute_size_preserving_conv_paddings(kernel_sizes):
    conv_paddings = [(int)((n - 1) / 2) for n in kernel_sizes]
    assert(all(padding==math.floor(padding) for padding in conv_paddings))
    return conv_paddings


#From the paper "Learning and Querying Generative Models for RL" (https://arxiv.org/pdf/1802.03006.pdf):
#Definition of the basic convolutional stack conv_stack: kernel size parameters k1, k2, k3
#  and channel parameters c1, c2, c3. Strides are always 1x1.
class ConvStack(nn.Module):
    def __init__(self,input_channels, kernel_sizes, output_channels):
        super(ConvStack, self).__init__()
        assert(len(kernel_sizes)==len(output_channels)==3)

        conv_paddings = compute_size_preserving_conv_paddings(kernel_sizes=kernel_sizes)

        # paddings are not given in the paper
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels[0], kernel_size=kernel_sizes[0], stride=1, padding=conv_paddings[0])
        self.conv2 = nn.Conv2d(in_channels=output_channels[0], out_channels=output_channels[1], kernel_size=kernel_sizes[1], stride=1, padding=conv_paddings[1])
        self.conv3 = nn.Conv2d(in_channels=output_channels[1], out_channels=output_channels[2], kernel_size=kernel_sizes[2], stride=1, padding=conv_paddings[2])

    def forward(self, x):
        intermediate_result = F.relu(self.conv1(x))
        x = self.conv2(intermediate_result)
        x = x + intermediate_result
        x = F.relu(x)
        x = self.conv3(x)
        return x


#The channel sizes and kernel sizes of this block are fixed in the paper.
class ResConv(nn.Module):
    def __init__(self, input_channels):
        super(ResConv, self).__init__()
        #paddings are not given in the paper, but it does not make sense otherwise (can't compute state + intermediate_result in forward)
        kernel_sizes = [3,5,3]
        self._output_channels = 64
        conv_paddings = compute_size_preserving_conv_paddings(kernel_sizes=kernel_sizes)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=kernel_sizes[0], stride=1, padding=conv_paddings[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_sizes[1], stride=1, padding=conv_paddings[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self._output_channels, kernel_size=kernel_sizes[2], stride=1, padding=conv_paddings[2])
        )

    def forward(self, state, input):
        intermediate_result = self.res_conv(input)
        output = state + intermediate_result
        return output

    def output_channels(self):
        return self._output_channels


# in "Learning and Querying Generative Models for RL" they used a size-preserving conv layer at the beginning of this module
class PoolAndInject(nn.Module):
    def __init__(self, input_channels, size):
        super(PoolAndInject, self).__init__()
        assert(len(size)==2)
        self.W, self.H = size
        #padding is not given in the paper, but it is not size-preserving otherwise
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(self.W, self.H))

    def forward(self,input):
        x = self.conv(input)
        # max-pool
        x = self.pool(x)
        # tile
        x = x.repeat(1, 1, self.W, self.H)
        # concat in channel dimension
        return torch.cat((x,input), 1)


class Flatten(torch.nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)

