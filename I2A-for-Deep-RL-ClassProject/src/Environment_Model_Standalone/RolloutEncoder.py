import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from I2A.ImaginationCore import ImaginationCore
import numpy as np

class EncoderCNNNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(EncoderCNNNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x, reward):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x


class EncoderLSTMNetwork(nn.Module):
    def __init__(self):
        super(EncoderLSTMNetwork, self).__init__()

        self.lstm_h = Variable(torch.zeros(1, 256)).type(FloatTensor)
        self.lstm_c = Variable(torch.zeros(1, 256)).type(FloatTensor)

        self.lstm = nn.LSTMCell(27200, 256, 1)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        print("F: ",x.data.shape)
        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h,self.lstm_c))
        x = self.lstm_h
        return x



class RolloutEncoder():
    def __init__(self, imagination_core, encoder_network, lstm_network, rollout_steps, start_action):
        self.imagination_core = imagination_core
        self.encoder_network = encoder_network
        self.lstm_network = lstm_network
        self.rollout_steps = rollout_steps
        self.start_action = start_action

    def imagine_future(self,input_state):
        imagined_states_and_rewards = []
        print("1. iteration of Rollout")
        next_state, reward = self.imagination_core.forward(input_state,self.start_action)
        imagined_states_and_rewards.append((next_state,reward))
        for i in range(self.rollout_steps-1):
            print((i+2),". iteration of Rollout")
            current_state = next_state
            next_state, reward = self.imagination_core.forward(current_state)
            imagined_states_and_rewards.append((next_state,reward))
        return imagined_states_and_rewards

    def encode(self, imagined_states_and_rewards):
        print("START ENCODE")
        cnn_outputs = []
        #reversed input for lstm, for cnn it doesnt matter which way
        print(len(imagined_states_and_rewards))
        for (state,reward) in reversed(imagined_states_and_rewards):
            cnn_output = self.encoder_network.forward(state,reward)
            broadcasted_reward = \
                Variable(torch.zeros(
                    cnn_output.data.shape[2],
                    cnn_output.data.shape[3],
                    cnn_output.data.shape[0])
                ).type(torch.FloatTensor) + \
                reward
            broadcasted_reward = torch.unsqueeze(broadcasted_reward.permute(2,0,1),0).cuda()
            aggregated_cnn_reward = torch.cat((cnn_output, broadcasted_reward), 1)
            print(aggregated_cnn_reward.data.shape)
            cnn_outputs.append(aggregated_cnn_reward)

        #only the last output of the LSTM is returned (the ones before are only to change the internal parameters?)
        for output in cnn_outputs:
            lstm_output = self.lstm_network.forward(output)
        return lstm_output[0]   #[0] because it is an [1x256] vector

    def forward(self,input_state):
        return self.encode(self.imagine_future(input_state))



use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

imagination_core = ImaginationCore(None,None)

input_channels_encoder = 1
encoder_network = EncoderCNNNetwork(input_channels_encoder)
encoder_network.cuda()
lstm_network = EncoderLSTMNetwork()
lstm_network.cuda()

#start_action = Variable(torch.from_numpy(np.array([0, 1]))).type(FloatTensor)
start_action = 0
rollout_encoder = RolloutEncoder(imagination_core, encoder_network, lstm_network, 3, start_action)

input_frame = np.ones(shape=(1,1,80,80))
input_frame = Variable(torch.from_numpy(input_frame)).type(FloatTensor)
res = rollout_encoder.forward(input_frame)
print(res)