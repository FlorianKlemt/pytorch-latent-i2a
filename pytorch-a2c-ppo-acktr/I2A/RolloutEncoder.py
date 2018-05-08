import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderCNNNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(EncoderCNNNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x


'''
Solved LSTM problems:
https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
https://github.com/pytorch/pytorch/issues/2769
https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226
'''
class EncoderLSTMNetwork(nn.Module):
    def __init__(self, number_lstm_cells, use_cuda=False):
        super(EncoderLSTMNetwork, self).__init__()

        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        # reward broadcasted = 6x6
        # lstm input = 6x6x64 + reward broadcast = 2340
        self.number_lstm_cells = number_lstm_cells

        #self.lstm = nn.LSTMCell(2520, self.number_lstm_cells, True)  # true for bias
        self.lstm = nn.LSTMCell(1700, self.number_lstm_cells, True)  # true for bias   10x10x16 + 1x10x10 (output size cnn + broadcasted reward)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        #self.fc = nn.Linear(12600, self.number_lstm_cells) #2340

    def forward(self,x):
        x = x.view(x.size(0), -1)

        # TODO hack ....
        if self.lstm_h.data.shape[0] != x.size(0) or self.lstm_c.data.shape[0] != x.size(0):
            print("Abort mission!")

        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h,self.lstm_c))
        x = self.lstm_h

        #x = self.fc(x)
        return x

    def repackage_lstm_hidden_variables(self, batch_size):
        #self.lstm_h = Variable(self.lstm_h.data)
        #self.lstm_c = Variable(self.lstm_c.data)
        self.lstm_h = Variable(torch.zeros(batch_size, self.number_lstm_cells)).type(self.FloatTensor)
        self.lstm_c = Variable(torch.zeros(batch_size, self.number_lstm_cells)).type(self.FloatTensor)


class RolloutEncoder():
    def __init__(self, imagination_core, encoder_network, lstm_network, rollout_steps, use_cuda):
        self.imagination_core = imagination_core
        self.encoder_network = encoder_network
        self.lstm_network = lstm_network
        self.rollout_steps = rollout_steps
        self.use_cuda = use_cuda
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def imagine_future(self,input_state, start_action):
        imagined_states = []
        imagined_rewards = []

        next_state, reward = self.imagination_core.forward(input_state, start_action)
        imagined_states.append(next_state)
        imagined_rewards.append(reward)

        for i in range(self.rollout_steps-1):
            current_state = next_state
            action = self.imagination_core.sample(current_state)
            next_state, reward = self.imagination_core.forward(current_state, action)
            imagined_states.append(next_state)
            imagined_rewards.append(reward)

        imagined_states = torch.stack(imagined_states, 1)
        imagined_rewards = torch.stack(imagined_rewards, 1)
        return imagined_states, imagined_rewards

    def encode(self, imagined_states_and_rewards):
        (states, rewards) = imagined_states_and_rewards

        shape = states.shape
        #change view forward and back for forwarding batchwise through cnn
        #order does not matter for cnn
        states = states.view(shape[0] * shape[1], shape[2], shape[3], shape[4])
        latent_space = self.encoder_network(states)
        latent_space = latent_space.view(shape[0], shape[1], latent_space.data.shape[1], latent_space.data.shape[2], latent_space.data.shape[3])

        broadcasted_reward = rewards.repeat(1, latent_space.data.shape[3], latent_space.data.shape[4], 1, 1)
        broadcasted_reward = broadcasted_reward.permute(3, 4, 0, 1, 2)

        aggregated = torch.cat((latent_space, broadcasted_reward), 2)
        # forward batchwise over the rollouts steps (permute to rollout_steps first, different action second)
        aggregated = aggregated.permute(1, 0, 2, 3, 4)
        # iterate in reverse order to feed data from last-to-first prediction into the LSTM
        for i in range(aggregated.data.shape[0]-1, -1, -1):
            lstm_input = aggregated[i].contiguous()
            lstm_output = self.lstm_network.forward(lstm_input)

        return lstm_output   #it is an [1x256] vector

    def forward(self, input_state, action):
        return self.encode(self.imagine_future(input_state, action))