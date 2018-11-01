import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from environment_model.mini_pacman.model.basic_blocks import BasicBlock
from model_helpers.flatten import Flatten
from model_helpers.model_initialization import xavier_weights_init_relu


class MiniPacmanEnvModel(torch.nn.Module):
    def __init__(self, obs_shape, num_actions, reward_bins, use_cuda):
        super(MiniPacmanEnvModel, self).__init__()
        self.num_actions = num_actions

        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        self.reward_bins = self.FloatTensor(reward_bins)

        input_channels = obs_shape[0]
        W=obs_shape[1]
        H=obs_shape[2]
        self.conv1 = nn.Conv2d(input_channels+self.num_actions, 64, kernel_size=1, stride=1, padding=0)    #input size is channels of input frame +1 for the broadcasted action
        self.basic_block1 = BasicBlock(num_inputs=64,n1=16,n2=32,n3=64,W=W,H=H,use_cuda=use_cuda)
        self.basic_block2 = BasicBlock(num_inputs=64, n1=16, n2=32, n3=64, W=W, H=H,use_cuda=use_cuda)
        #self.basic_block2 = BasicBlock(num_inputs=128, n1=16, n2=32, n3=64, W=W, H=H, use_cuda=use_cuda)
        self.reward_head = nn.Sequential(OrderedDict([
          #('reward_conv1', nn.Conv2d(192, 64, kernel_size=1)),  # input size is n3 of basic-block2
          ('reward_conv1', nn.Conv2d(64, 64, kernel_size=1)),  #input size is n3 of basic-block2
          ('reward_relu1', nn.ReLU()),
          ('reward_conv2', nn.Conv2d(64, 64, kernel_size=1)),
          ('reward_relu2', nn.ReLU()),
          ('flatten',      Flatten()),
          ('reward_fc',    nn.Linear(64*W*H, 5)),
          ('softmax',      nn.Softmax(dim=1))
        ]))
        self.img_head = nn.Sequential(OrderedDict([
            #('conv1', nn.Conv2d(192, input_channels, kernel_size=1)),
            ('conv1',      nn.Conv2d(64, input_channels, kernel_size=1)),        #input size is n3 of basic-block2, output is input_channels (1 or 3)
            ('sigmoid',    nn.Sigmoid())
        ]))

        self.apply(xavier_weights_init_relu)

        self.train()

    def forward(self,input_frame,input_action):
        one_hot = torch.zeros(input_action.shape[0], self.num_actions).type(self.FloatTensor)
        # make one hot vector
        one_hot.scatter_(1, input_action, 1)
        # broadcast action
        one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)
        broadcasted_action = one_hot.repeat(1, 1, input_frame.shape[2], input_frame.shape[3])

        # concatinate observation and broadcasted action
        x = torch.cat((input_frame,broadcasted_action),1)

        x = F.relu(self.conv1(x))
        x = self.basic_block1(x)
        x = self.basic_block2(x)

        #output image head
        image_out = self.img_head(x)

        #output reward head
        reward_probability = self.reward_head(x)
        reward_out = torch.sum(reward_probability * self.reward_bins, 1)

        return image_out,reward_out

