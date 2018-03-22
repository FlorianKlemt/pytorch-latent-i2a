import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class MiniPacManEM(nn.Module):

    def __init__(self, input_channels=3, output_channels=2):
        super(MiniPacManEM, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=8)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        #NOTE: needs 1 as output channels otherwise the state_size grows with each rollout iteration
        self.deconv = nn.ConvTranspose2d(32*2, 1, kernel_size=8, stride=8)

        self.conv_r1 = nn.Conv2d(32*2, 32, kernel_size=3, stride=1, padding=1)
        self.conv_r2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        linear_in_size = 128 #TODO formel finden
        self.fc_r1 = nn.Linear(linear_in_size, output_channels)

    def forward(self, input_frame, input_action):   #input_action should be one_hot
        print(type(input_action))
        print(input_action.data.shape, input_frame.data.shape)
        print(input_action)

        input_action = \
            Variable(torch.zeros(
                input_frame.data.shape[2],
                input_frame.data.shape[3],
                input_action.data.shape[0])
            ).type(FloatTensor) + \
            input_action
        #input_action = np.broadcast_to(input_action.data.numpy(), (80, 80, input_action.data.shape[0]))
        #input_action = Variable(torch.from_numpy(input_action)).type(FloatTensor)
        print ("Input_action",input_action.data[0][0][:])
        print(input_action.data.shape, input_frame.data.shape)
        input_action = torch.unsqueeze(input_action, 0).permute(0,3,1,2)
        print(input_action.data.shape, input_frame.data.shape)
        x = torch.cat([input_frame, input_action], 1)

        x = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x2))
        #skip connection
        x = torch.cat((x,x2),1)

        #compute output image
        output_image = F.relu(self.deconv(x))

        #compute output reward
        #print("shape x: ",x.data.shape)
        output_reward = F.leaky_relu(self.conv_r1(x))
        output_reward = F.max_pool2d(output_reward, kernel_size=2, stride=2, padding=0)
        output_reward = F.leaky_relu(self.conv_r2(output_reward))
        output_reward = F.max_pool2d(output_reward, kernel_size=2, stride=2, padding=0)
        output_reward = output_reward.view(output_reward.size(0), -1)
        output_reward = F.softmax(self.fc_r1(output_reward))

        print("Env output shape: ",output_image.data.shape)
        return output_image, output_reward

use_cuda = torch.cuda.is_available()
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

input_frame = np.ones(shape=(1,1,80,80))
input_frame = Variable(torch.from_numpy(input_frame)).type(FloatTensor)

DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

input_frame = np.ones(shape=(1,1,80,80))
input_frame = Variable(torch.from_numpy(input_frame)).type(FloatTensor)

input_action = Variable(torch.from_numpy(np.array([0, 1]))).type(FloatTensor)

net = EnvNetwork()
net.cuda()
print(net.forward(input_frame,input_action))