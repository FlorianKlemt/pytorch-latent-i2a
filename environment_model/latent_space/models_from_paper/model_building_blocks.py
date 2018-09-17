import torch
import torch.nn as nn
import torch.nn.functional as F
from environment_model.latent_space.models_from_paper.depth2space import DepthToSpace, SpaceToDepth
from i2a.utils import get_linear_dims_after_conv
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


#Transition Module for computing the state transition function s_t=g(s_t-1, z_t, a_t-1)
#Note: the output shape of the state transition has to be the same as the input shape
class StateTransition(nn.Module):
    def __init__(self, input_shape, num_actions, use_stochastic, use_cuda):
        super(StateTransition, self).__init__()
        self.use_cuda = use_cuda
        self.num_actions = num_actions
        self.use_stochastic = use_stochastic
        self.input_shape = input_shape
        if use_stochastic:
            # state channels + channels of broadcasted action + broadcasted z
            input_channels = input_shape[0] + num_actions + input_shape[0]
        else:
            input_channels = input_shape[0] + num_actions

        self.res_conv1 = ResConv(input_channels=input_channels)
        self.pool_and_inject = PoolAndInject(input_channels=self.res_conv1.output_channels(), size=(input_shape[1],input_shape[2]))
        # pool-and-inject output channels should be nr input channels + 32
        self.res_conv2 = ResConv(input_channels=96)

    def forward(self, state, action, z):
        broadcasted_action = broadcast_action(action=action, num_actions=self.num_actions, broadcast_to_shape=state.shape[2:], use_cuda=self.use_cuda)

        if self.use_stochastic and z is not None:
            concatenated = torch.cat((state, broadcasted_action, z), 1)
        else:
            concatenated = torch.cat((state, broadcasted_action), 1)

        x = F.relu(self.res_conv1(state, concatenated))
        x = self.pool_and_inject(x)
        x = self.res_conv2(state,x)
        return x

    def output_size(self):
        return self.input_shape


# All input dimension sizes (widht, height) need to be divisible by
# first_space_to_depth_block_size*second_space_to_depth_block_size (8 in our case)
class EncoderModule(nn.Module):
    def __init__(self, input_shape):
        super(EncoderModule, self).__init__()
        self.input_shape = input_shape
        first_space_to_depth_block_size = 4
        conv_stack1_input_channels = input_shape[0] * pow(first_space_to_depth_block_size, 2)
        conv_stack1_output_channels = 64
        second_space_to_depth_block_size = 2
        conv_stack2_input_channels = conv_stack1_output_channels * pow(second_space_to_depth_block_size, 2)
        self.output_channels = 64
        self.division_size = first_space_to_depth_block_size * second_space_to_depth_block_size
        assert (math.fmod(input_shape[1], self.division_size)==0 and math.fmod(input_shape[2], self.division_size)==0),\
            "Input Dimensions "+input_shape[1]+"x"+input_shape[2]+" need to be divisible by "+self.division_size
        self.encoder = nn.Sequential(
            SpaceToDepth(block_size=first_space_to_depth_block_size),
            ConvStack(input_channels=conv_stack1_input_channels, kernel_sizes=(3,5,3), output_channels=(16,16,conv_stack1_output_channels)),
            SpaceToDepth(block_size=second_space_to_depth_block_size),
            ConvStack(input_channels=conv_stack2_input_channels, kernel_sizes=(3,5,3), output_channels=(32,32,self.output_channels)),
            nn.ReLU()
        )

    def forward(self, observation):
        encoded = self.encoder(observation)
        return encoded

    def output_size(self):
        return (self.output_channels,self.input_shape[1]/self.division_size, self.input_shape[2]/self.division_size)



class Flatten(torch.nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)



class DecoderModule(nn.Module):
    def __init__(self, input_shape, use_vae, reward_prediction_bits):
        super(DecoderModule, self).__init__()
        self.use_vae = use_vae
        reward_head_conv = nn.Conv2d(in_channels=input_shape[0], out_channels=24, kernel_size=3, stride=1)
        reward_head_linear_dims = get_linear_dims_after_conv([reward_head_conv], (input_shape[1], input_shape[2]))
        self.reward_head = nn.Sequential(
            reward_head_conv,
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=reward_head_linear_dims, out_features=reward_prediction_bits)
        )

        if self.use_vae:
            #state input channels + channels for z
            input_channels = input_shape[0] + input_shape[0]
        else:
            input_channels = input_shape[0]

        image_head_conv1_output_channels = 64
        image_head_d2s1_block_size = 2
        image_head_conv2_input_channels = image_head_conv1_output_channels/(pow(image_head_d2s1_block_size, 2))
        self.image_head = nn.Sequential(
            ConvStack(input_channels=input_channels, kernel_sizes=(1, 5, 3), output_channels=(32, 32, image_head_conv1_output_channels)),
            DepthToSpace(block_size=image_head_d2s1_block_size),
            ConvStack(input_channels=image_head_conv2_input_channels, kernel_sizes=(3, 3, 1), output_channels=(64, 64, 48)),
            DepthToSpace(block_size=4)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, state, z):
        reward_log_probs = self.reward_head(state)

        if self.use_vae and z is not None:
            concatenated = torch.cat((state, z), 1)
        else:
            concatenated = state

        image_log_probs = self.image_head(concatenated)
        image_log_probs = self.sigmoid(image_log_probs)
        return image_log_probs, reward_log_probs



#Prior Module for computing mean μ_z_t and diagonal variance σ_z_t of the normal distribution p(z_t|s_t-1, a_t-1)
class PriorModule(nn.Module):
    def __init__(self, state_input_channels, num_actions, use_cuda):
        super(PriorModule, self).__init__()
        self.use_cuda = use_cuda
        self.num_actions = num_actions
        input_channels = state_input_channels + num_actions
        self.conv_stack = ConvStack(input_channels=input_channels, kernel_sizes=(1,3,3), output_channels=(32,32,64))

    #inputs are s_t-1 and a_t-1
    def forward(self, state, action):
        assert(len(state.shape)==4)
        broadcasted_action = broadcast_action(action=action, num_actions=self.num_actions, broadcast_to_shape=state.shape[-2:], use_cuda=self.use_cuda)
        concatenated = torch.cat((state, broadcasted_action), 1)

        mu = self.conv_stack(concatenated)

        sigma = torch.log(1 + torch.exp(mu))
        return mu, sigma


#Posterior Module for computing mean μ'_z_t and diagonal variance σ'_z_t of the normal distribution q(z_z|s_t-1, a_t-1, o_t).
#   The posterior gets as additional inputs the prior statistics μ_z_t, σ_z_t.
class PosteriorModule(nn.Module):
    def __init__(self, state_input_channels, num_actions, use_cuda):
        super(PosteriorModule, self).__init__()
        self.use_cuda = use_cuda
        self.num_actions = num_actions
        #Explanation: input channels = state channels + action broadcast channels
        #                               + 64 channels encoded observation + 64 channel broadcasted mu + 64 channel broadcasted sigma
        input_channels = state_input_channels + num_actions + 64 + 64 + 64
        self.conv_stack = ConvStack(input_channels=input_channels, kernel_sizes=(1,3,3), output_channels=(32,32,64))

    def forward(self, prev_state, action, encoded_obs, mu, sigma):
        broadcasted_action = broadcast_action(action=action, num_actions=self.num_actions, broadcast_to_shape=prev_state.shape[2:], use_cuda=self.use_cuda)
        concatenated = torch.cat((prev_state, broadcasted_action, encoded_obs, mu, sigma), 1)

        mu_posterior = self.conv_stack(concatenated)

        sigma_posterior = torch.log(1 + torch.exp(mu_posterior))
        return mu_posterior, sigma_posterior




class InitialStateModule(nn.Module):
    def __init__(self):
        super(InitialStateModule, self).__init__()
        # encoded states of o_0, o_-1, o_-2 each 64 channels
        input_channels = 64 + 64 + 64
        self.conv_stack = ConvStack(input_channels=input_channels, kernel_sizes=(1,3,3), output_channels=(64,64,64))

    def forward(self, encoded_now, encoded_prev, encoded_2prev):
        concatenated = torch.cat((encoded_now, encoded_prev, encoded_2prev), 1)
        x = self.conv_stack(concatenated)
        return x
