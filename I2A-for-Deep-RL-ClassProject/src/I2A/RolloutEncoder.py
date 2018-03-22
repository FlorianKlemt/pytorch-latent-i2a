import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EncoderCNNNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(EncoderCNNNetwork, self).__init__()

        #self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0) # 19x19x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0) # 8x8x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0) # 6x6x64


    def forward(self, x, reward):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        return x


'''
TODO: !!!
LSTM does not work because it needs a backword pass where 
retain_graph is set to True, but it looks like this produce
an memory leak on the gpu.... and we run out of memory very fast!!
As a work around (not sure if it will learn the right thing) 
we replaced the lstm with an fully connected layer

solved:
https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
https://github.com/pytorch/pytorch/issues/2769
https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226
'''
class EncoderLSTMNetwork(nn.Module):
    def __init__(self, number_lstm_cells, use_cuda=False):
        super(EncoderLSTMNetwork, self).__init__()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        # reward broadcasted = 6x6
        # lstm input = 6x6x64 + reward broadcast = 2340
        self.number_lstm_cells = number_lstm_cells


        self.lstm_h = Variable(torch.zeros(1, self.number_lstm_cells)).type(FloatTensor)
        self.lstm_c = Variable(torch.zeros(1, self.number_lstm_cells)).type(FloatTensor)

        #self.lstm = nn.LSTMCell(2340, self.number_lstm_cells, True) #true for bias
        self.lstm = nn.LSTMCell(2520, self.number_lstm_cells, True)  # true for bias
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)


        #self.fc = nn.Linear(12600, self.number_lstm_cells) #2340

    def forward(self,x):
        x = x.view(x.size(0), -1)

        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h,self.lstm_c))
        x = self.lstm_h

        #x = self.fc(x)
        return x

    def repackage_lstm_hidden_variables(self):
        self.lstm_h = Variable(self.lstm_h.data)
        self.lstm_c = Variable(self.lstm_c.data)


class RolloutEncoder():
    def __init__(self, imagination_core, encoder_network, lstm_network, rollout_steps, start_action, use_cuda):
        self.imagination_core = imagination_core
        self.encoder_network = encoder_network
        self.lstm_network = lstm_network
        self.rollout_steps = rollout_steps
        self.start_action = start_action
        self.use_cuda = use_cuda
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def imagine_future(self,input_state):
        imagined_states_and_rewards = []
        next_state, reward = self.imagination_core.forward(input_state,self.start_action)
        imagined_states_and_rewards.append((next_state,reward))
        for i in range(self.rollout_steps-1):
            current_state = next_state
            next_state, reward = self.imagination_core.forward(current_state)
            imagined_states_and_rewards.append((next_state,reward))
        return imagined_states_and_rewards

    def encode(self, imagined_states_and_rewards):
        cnn_outputs = []
        #reversed input for lstm, for cnn it doesnt matter which way
        for (state,reward) in reversed(imagined_states_and_rewards):
            cnn_output = self.encoder_network.forward(state,reward)
            broadcasted_reward = \
                Variable(torch.zeros(
                    cnn_output.data.shape[2],
                    cnn_output.data.shape[3],
                    cnn_output.data.shape[0])
                ).type(self.FloatTensor) + \
                reward
            broadcasted_reward = torch.unsqueeze(broadcasted_reward.permute(2,0,1),0)
            if self.use_cuda:
                broadcasted_reward = broadcasted_reward.cuda()
            aggregated_cnn_reward = torch.cat((cnn_output, broadcasted_reward), 1)
            cnn_outputs.append(aggregated_cnn_reward)

        #only the last output of the LSTM is returned (the ones before are only to change the internal parameters?)
        ''' TODO: comment out test fully connected version '''
        #cnn_outputs = torch.cat(cnn_outputs, 1)
        #lstm_output = self.lstm_network(cnn_outputs)
        ''' TODO end'''

        ''' TODO: comment in orginal lstm version '''
        for output in cnn_outputs:
            lstm_output = self.lstm_network.forward(output)
        ''' TODO end'''

        return lstm_output   #it is an [1x256] vector

    def forward(self,input_state):
        return self.encode(self.imagine_future(input_state))

    def repackage_lstm_hidden_variables(self):
        self.imagination_core.repackage_lstm_hidden_variables()
        self.lstm_network.repackage_lstm_hidden_variables()