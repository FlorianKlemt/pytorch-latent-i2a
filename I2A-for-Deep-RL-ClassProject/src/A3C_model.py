from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init
from torch.autograd import Variable

class A3Clstm(torch.nn.Module):
    """
    A3Clstm
    """
    #Note: the use_cuda parameter has no actual use here but it makes it easier to switch between architectures, has they have the same paramters
    def __init__(self, num_inputs, action_space, use_cuda):
        super(A3Clstm, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(1024, 512)


        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512,action_space)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        self.hx = Variable(torch.zeros(1, 512), requires_grad=False).type(FloatTensor)
        self.cx = Variable(torch.zeros(1, 512), requires_grad=False).type(FloatTensor)

        self.train()


    def forward(self, inputs):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))

        x = self.hx

        policy = self.actor_linear(x)
        value = self.critic_linear(x)
        return value, policy

    def repackage_lstm_hidden_variables(self):
        self.hx = Variable(self.hx.data)
        self.cx = Variable(self.cx.data)

class SmallA3Clstm(torch.nn.Module):
    #Note: the use_cuda parameter has no actual use here but it makes it easier to switch between architectures, has they have the same paramters
    def __init__(self, num_inputs, action_space, use_cuda):
        super(SmallA3Clstm, self).__init__()
        self.num_actions = action_space

        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4, padding=0) # 19x19x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0) # 8x8x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0) # 6x6x64

        #self.fc = nn.Linear(2304, num_outputs)

        self.lstm = nn.LSTMCell(2304, 512)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.hx = Variable(torch.zeros(1, 512)).type(self.FloatTensor)
        self.cx = Variable(torch.zeros(1, 512)).type(self.FloatTensor)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512,action_space)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))

        x = self.hx

        policy = self.actor_linear(x)
        value = self.critic_linear(x)
        return value, policy

    def repackage_lstm_hidden_variables(self):
        self.hx = Variable(self.hx.data)
        self.cx = Variable(self.cx.data)


