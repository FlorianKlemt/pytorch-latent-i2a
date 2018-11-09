import torch
import torch.nn as nn
import torch.nn.functional as F

from model_helpers.model_initialization import xavier_weights_init_relu


# in "Imagination Augmented Agents" they NO conv layer at the beginning of this module
class PoolAndInject(torch.nn.Module):
    def __init__(self,W,H,use_cuda):
        super(PoolAndInject, self).__init__()
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.W = W
        self.H = H
        self.pool = nn.MaxPool2d(kernel_size=(W,H))

    def forward(self,input):
        x = F.relu(self.pool(input))   #max-pool
        x = x.repeat(1, 1, self.W, self.H)  #tile
        return x + input


class BasicBlock(torch.nn.Module):
    def __init__(self,num_inputs,n1,n2,n3,W,H,use_cuda):
        super(BasicBlock, self).__init__()
        self.pool_and_inject = PoolAndInject(W,H,use_cuda)

        input_channels = num_inputs
        # pool-and-inject layer is size-preserving therefore num_inputs is the input to conv1
        self.left_conv1 = nn.Conv2d(input_channels, n1, kernel_size=1)
        self.left_conv2 = nn.Conv2d(n1, n1, kernel_size=9, padding=4)
        self.right_conv1 = nn.Conv2d(input_channels, n2, kernel_size=1)
        self.right_conv2 = nn.Conv2d(n2, n2, kernel_size=3, padding=1)
        # input after cat is output size of left side + output size of right side = n1 + n2
        self.conv3 = nn.Conv2d(n1+n2, n3, kernel_size=1)
        self.apply(xavier_weights_init_relu)

    def forward(self,input):
        x = self.pool_and_inject(input)
        left_side = F.relu(self.left_conv2(F.relu(self.left_conv1(x))))
        right_side = F.relu(self.right_conv2(F.relu(self.right_conv1(x))))
        x = torch.cat((left_side,right_side),1)
        x = F.relu(self.conv3(x))
        return x + input
