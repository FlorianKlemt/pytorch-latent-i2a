import torch
import torch.nn as nn
import torch.nn.functional as F
from i2a.utils import get_linear_dims_after_conv, get_conv_output_dims
from functools import reduce

class LatentSpaceEncoderCNNNetwork(nn.Module):
    def __init__(self, encoding_shape):
        super(LatentSpaceEncoderCNNNetwork, self).__init__()
        input_channels = encoding_shape[0]
        input_dims = encoding_shape[1:]

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

        self.output_dims = reduce(lambda x, y: x * y, get_conv_output_dims([self.conv1, self.conv2], input_dims))
        self.output_size = self.conv2.out_channels * self.output_dims

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x



class LatentSpaceEncoderLSTMNetwork(nn.Module):
    def __init__(self, input_dim, number_lstm_cells, use_cuda=False):
        super(LatentSpaceEncoderLSTMNetwork, self).__init__()

        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.use_cuda = use_cuda

        self.number_lstm_cells = number_lstm_cells

        # input_dim = (output size cnn + broadcasted reward)
        self.lstm = nn.LSTMCell(input_dim, self.number_lstm_cells, True)

    def forward(self,x):
        x = x.view(x.size(0), -1)

        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h,self.lstm_c))
        x = self.lstm_h

        return x

    def repackage_lstm_hidden_variables(self, batch_size):
        if self.use_cuda:
            self.lstm_h = torch.cuda.FloatTensor(batch_size, self.number_lstm_cells).fill_(0)
            self.lstm_c = torch.cuda.FloatTensor(batch_size, self.number_lstm_cells).fill_(0)
        else:
            self.lstm_h = torch.FloatTensor(batch_size, self.number_lstm_cells).fill_(0)
            self.lstm_c = torch.FloatTensor(batch_size, self.number_lstm_cells).fill_(0)


class LatentSpaceRolloutEncoder():
    def __init__(self, imagination_core, encoder_network, lstm_network, rollout_steps, use_cuda):
        self.imagination_core = imagination_core
        self.encoder_network = encoder_network
        self.lstm_network = lstm_network
        self.rollout_steps = rollout_steps
        self.use_cuda = use_cuda
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def imagine_future(self,input_latent_space, start_action):
        next_latent_state, z_prior, reward = self.imagination_core.forward(input_latent_space, start_action)
        imagined_states = next_latent_state.unsqueeze(1)
        imagined_rewards = reward.unsqueeze(1)

        for i in range(self.rollout_steps-1):
            current_latent_state = next_latent_state
            action = self.imagination_core.sample(current_latent_state)
            next_latent_state, z_prior, reward = self.imagination_core.forward(current_latent_state, action)

            imagined_states = torch.cat((imagined_states, next_latent_state.unsqueeze(1)), dim=1)
            imagined_rewards = torch.cat((imagined_rewards, reward.unsqueeze(1)), dim=1)

        return imagined_states, imagined_rewards

    def encode(self, imagined_states_and_rewards):
        (states, rewards) = imagined_states_and_rewards

        shape = states.shape
        #change view forward and back for forwarding batchwise through cnn
        #order does not matter for cnn
        states = states.view(shape[0] * shape[1], shape[2], shape[3], shape[4])
        latent_space = self.encoder_network(states)
        latent_space = latent_space.view(shape[0], shape[1], latent_space.shape[1], latent_space.shape[2], latent_space.shape[3])

        broadcasted_reward = rewards.view(rewards.shape[0], rewards.shape[1],1,1,1)\
                                     .repeat(1, 1, 1, latent_space.shape[3], latent_space.shape[4])

        aggregated = torch.cat((latent_space, broadcasted_reward), 2)
        # forward batchwise over the rollouts steps (permute to rollout_steps first, different action second)
        aggregated = aggregated.permute(1, 0, 2, 3, 4)
        # iterate in reverse order to feed data from last-to-first prediction into the LSTM
        for i in range(aggregated.shape[0]-1, -1, -1):
            lstm_input = aggregated[i].contiguous()
            lstm_output = self.lstm_network.forward(lstm_input)

        return lstm_output   #it is an [batchsizex256] vector

    def forward(self, input_state, action):
        return self.encode(self.imagine_future(input_state, action))