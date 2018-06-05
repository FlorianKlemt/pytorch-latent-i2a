import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


from I2A.EnvironmentModel.minipacman_rgb_class_converter import MiniPacmanRGBToClassConverter
import torch.nn.init as init
import math

def xavier_weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        m.weight.data.mul_(math.sqrt(2))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Flatten(torch.nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)

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
        self.left_conv1 = nn.Conv2d(num_inputs, n1, kernel_size=1)  #pool-and-inject layer is size-preserving therefore num_inputs is the input to conv1
        self.left_conv2 = nn.Conv2d(n1, n1, kernel_size=9, padding=4)
        self.right_conv1 = nn.Conv2d(num_inputs, n2, kernel_size=1)
        self.right_conv2 = nn.Conv2d(n2, n2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n1+n2, n3, kernel_size=1)            #input after cat is output size of left side + output size of right side = n1 + n2
        self.apply(xavier_weights_init)

    def forward(self,input):
        x = self.pool_and_inject(input)
        left_side = F.relu(self.left_conv2(F.relu(self.left_conv1(x))))
        right_side = F.relu(self.right_conv2(F.relu(self.right_conv1(x))))
        x = torch.cat((left_side,right_side),1)
        x = F.relu(self.conv3(x))
        return x + input

class MiniPacmanEnvModelClassLabels(torch.nn.Module):
    def __init__(self, obs_shape, num_actions, reward_bins, use_cuda):
        super(MiniPacmanEnvModelClassLabels, self).__init__()
        self.num_actions = num_actions

        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        self.reward_bins = Variable(torch.FloatTensor(reward_bins).type(self.FloatTensor))

        input_channels = obs_shape[0]
        W=obs_shape[1]
        H=obs_shape[2]
        self.conv1 = nn.Conv2d(input_channels+self.num_actions, 64, kernel_size=1, stride=1, padding=0)    #input size is channels of input frame +1 for the broadcasted action
        self.basic_block1 = BasicBlock(num_inputs=64,n1=16,n2=32,n3=64,W=W,H=H,use_cuda=use_cuda)
        self.basic_block2 = BasicBlock(num_inputs=64, n1=16, n2=32, n3=64, W=W, H=H,use_cuda=use_cuda)
        self.reward_head = nn.Sequential(OrderedDict([
          ('reward_conv1', nn.Conv2d(64, 64, kernel_size=1)),  #input size is n3 of basic-block2
          ('reward_relu1', nn.ReLU()),
          ('reward_conv2', nn.Conv2d(64, 64, kernel_size=1)),
          ('reward_relu2', nn.ReLU()),
          ('flatten',      Flatten()),
          # TODO why do they use 5 output rewards??
          #('reward_fc', nn.Linear(64 * W * H, 1))
          ('reward_fc',    nn.Linear(64*W*H, 5))#,
          #('relu',         nn.ReLU())
        ]))

        self.img_head = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, input_channels, kernel_size=1))#,
            #('softmax', nn.Softmax())
        ]))

        self.apply(xavier_weights_init)

        self.train()

        self.rgb_to_class = MiniPacmanRGBToClassConverter(use_cuda=use_cuda)

    def forward(self,input_frame,input_action):
        input_frame = self.rgb_to_class.minipacman_rgb_to_class(input_frame)

        image_out, reward_out = self.forward_class(input_frame, input_action)

        image_out = self.rgb_to_class.minipacman_class_to_rgb(image_out)

        return image_out,reward_out

    def forward_class(self,input_frame,input_action):
        one_hot = torch.zeros(input_action.data.shape[0], self.num_actions).type(self.FloatTensor)
        # make one hot vector
        one_hot.scatter_(1, input_action.data, 1)
        # breoadcast action
        one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)
        broadcasted_action = Variable(one_hot.repeat(1, 1, input_frame.data.shape[2], input_frame.data.shape[3]))

        # concatinate observation and broadcasted action
        x = torch.cat((input_frame,broadcasted_action),1)

        x = F.relu(self.conv1(x))
        x = self.basic_block1(x)
        x = self.basic_block2(x)

        #output image head
        image_out = self.img_head(x)

        #output reward head
        reward_probability = self.reward_head(x)
        # TODO why not just use the value of the max probability
        # TODO is this correct??
        reward_out = torch.sum(reward_probability * self.reward_bins, 1)
        #x.view(x.size())

        return image_out,reward_out





class MiniPacmanEnvModel(torch.nn.Module):
    def __init__(self, obs_shape, num_actions, reward_bins, use_cuda):
        super(MiniPacmanEnvModel, self).__init__()
        self.num_actions = num_actions

        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        self.reward_bins = Variable(torch.FloatTensor(reward_bins).type(self.FloatTensor))

        input_channels = obs_shape[0]
        W=obs_shape[1]
        H=obs_shape[2]
        self.conv1 = nn.Conv2d(input_channels+self.num_actions, 64, kernel_size=1, stride=1, padding=0)    #input size is channels of input frame +1 for the broadcasted action
        self.basic_block1 = BasicBlock(num_inputs=64,n1=16,n2=32,n3=64,W=W,H=H,use_cuda=use_cuda)
        self.basic_block2 = BasicBlock(num_inputs=64, n1=16, n2=32, n3=64, W=W, H=H,use_cuda=use_cuda)
        self.reward_head = nn.Sequential(OrderedDict([
          ('reward_conv1', nn.Conv2d(64, 64, kernel_size=1)),  #input size is n3 of basic-block2
          ('reward_relu1', nn.ReLU()),
          ('reward_conv2', nn.Conv2d(64, 64, kernel_size=1)),
          ('reward_relu2', nn.ReLU()),
          ('flatten',      Flatten()),
          # TODO why do they use 5 output rewards??
          #('reward_fc', nn.Linear(64 * W * H, 1))
          ('reward_fc',    nn.Linear(64*W*H, 5))#,
          #('softmax',      nn.Softmax())
        ]))
        self.img_head = nn.Conv2d(64, input_channels, kernel_size=1)        #input size is n3 of basic-block2, output is input_channels (1 or 3)

        self.apply(xavier_weights_init)

        #TODO: this formulation destroys compatibility with current I2A implementation
        #self.img_head = nn.Sequential(OrderedDict([
        #    ('conv', nn.Conv2d(64, input_channels, kernel_size=1))#,
        #    #('softmax', nn.Softmax())
        #]))
        #torch.nn.init.xavier_uniform(self.img_head.conv.weight)

        self.train()

    def forward(self,input_frame,input_action):
        one_hot = torch.zeros(input_action.data.shape[0], self.num_actions).type(self.FloatTensor)
        # make one hot vector
        one_hot.scatter_(1, input_action.data, 1)
        # breoadcast action
        one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)
        broadcasted_action = Variable(one_hot.repeat(1, 1, input_frame.data.shape[2], input_frame.data.shape[3]))

        # concatinate observation and broadcasted action
        x = torch.cat((input_frame,broadcasted_action),1)

        x = F.relu(self.conv1(x))
        x = self.basic_block1(x)
        x = self.basic_block2(x)

        #output image head
        image_out = self.img_head(x)

        #output reward head
        reward_probability = self.reward_head(x)
        # TODO why not just use the value of the max probability
        # TODO is this correct??
        reward_out = torch.sum(reward_probability * self.reward_bins, 1)

        return image_out,reward_out


#the copy model returns the identity, this is its own class so we dont have to change the code to use the copymodel
class CopyEnvModel(torch.nn.Module):
    def __init__(self):
        super(CopyEnvModel, self).__init__()
    def forward(self, input_frame, input_action):
        return input_frame, Variable(torch.zeros(input_frame.data.shape[0])).cuda()