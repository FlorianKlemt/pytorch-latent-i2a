import torch
import torch.nn as nn
import torch.nn.functional as F
from LatentSpaceEncoder.models_from_paper.depth2space import DepthToSpace, SpaceToDepth



def broadcast_action(action, num_actions, broadcast_to_shape, use_cuda):
    assert(len(broadcast_to_shape)==2)
    one_hot = torch.zeros(action.shape[0], num_actions)
    if use_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, action, 1)  # make one-hot vector
    one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)
    broadcasted_action = one_hot.repeat(1, 1, *broadcast_to_shape)  # broadcast to width and height of broadcast_to_shape
    return broadcasted_action


#From the Paper:
#Definition of the basic convolutional stack conv_stack: kernel size parameters k1, k2, k3
#  and channel parameters c1, c2, c3. Strides are always 1x1.
class ConvStack(nn.Module):
    def __init__(self,input_channels, kernel_sizes, output_channels):
        super(ConvStack, self).__init__()
        assert(len(kernel_sizes)==len(output_channels)==3)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels[0], kernel_size=kernel_sizes[0], stride=1)
        self.conv2 = nn.Conv2d(in_channels=output_channels[0], out_channels=output_channels[1], kernel_size=kernel_sizes[1], stride=1, padding=2) #padding 2 is not in the paper but it doesnt seem to work otherwise (at least for 84,84 obs size)
        self.conv3 = nn.Conv2d(in_channels=output_channels[1], out_channels=output_channels[2], kernel_size=kernel_sizes[2], stride=1)

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
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        )

    def forward(self, x):
        intermediate_result = self.res_conv(x)
        output = x + intermediate_result
        return output


class PoolAndInject(nn.Module):
    def __init__(self, input_channels, size):
        super(PoolAndInject, self).__init__()
        assert(len(size)==2)
        self.W, self.H = size
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=(self.W, self.H))

    def forward(self,input):
        x = self.conv(input)
        x = self.pool(x)   #max-pool
        x = x.repeat(1, 1, self.W, self.H)  #tile
        return torch.cat((x,input), 1)  # concat in channel dimension


#Transition Module for computing the state transition function s_t=g(s_t-1, z_t, a_t-1)
class StateTransition(nn.Module):
    def __init__(self, state_input_channels, num_actions, use_cuda):
        super(StateTransition, self).__init__()
        self.use_cuda = use_cuda
        self.num_actions = num_actions  #needed to broadcast the action (they are broadcasted to as many channels as there are actions)
        input_channels = state_input_channels + num_actions + 1 #state channels + channels of broadcasted action + 1 channel of broadcasted z
        self.transition = nn.Sequential(
            ResConv(input_channels=input_channels),
            nn.ReLU(),
            PoolAndInject(input_channels=64, size=(10,10)), #TODO: size is wrong!! placeholder  #output channels of res_conv should always be 64 channels
            ResConv(input_channels=96) #pool-and-inject output channels should be nr input channels + 32
        )

    def forward(self, state, action, z):
        broadcasted_action = broadcast_action(action=action, num_actions=self.num_actions, broadcast_to_shape=state.shape[2:], use_cuda=self.use_cuda)

        #we broadcast z to 1 channel --> TODO: check if this is correct
        broadcasted_z = z.repeat(1, 1, state.shape[2], state.shape[3])  #assumes z is given as a torch tensor

        # concatinate observation and broadcasted action
        concatenated = torch.cat((state, broadcasted_action, broadcasted_z), 1)

        x = self.transition(concatenated)
        return x



class EncoderModule(nn.Module):
    def __init__(self, input_channels):
        super(EncoderModule, self).__init__()
        first_space_to_depth_block_size = 4
        self.encoder = nn.Sequential(
            SpaceToDepth(block_size=first_space_to_depth_block_size),
            ConvStack(input_channels=input_channels*pow(first_space_to_depth_block_size,2), kernel_sizes=(3,5,3), output_channels=(16,16,64)),
            SpaceToDepth(block_size=2),
            ConvStack(input_channels=1, kernel_sizes=(3,5,3), output_channels=(32,32,64)),  #TODO: input channels are wrong!! placeholder
            nn.ReLU()
        )

    def forward(self, observation):
        encoded = self.encoder(observation)
        return encoded



class Flatten(torch.nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)



class DecoderModule(nn.Module):
    def __init__(self, state_input_channels):
        super(DecoderModule, self).__init__()
        self.reward_head = nn.Sequential(
            nn.Conv2d(in_channels=state_input_channels, out_channels=24, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),  #the paper says only reshape, but should be a Flatten as it is followed by a Linear layer
            nn.Linear(in_features=1, out_features=1)    #TODO: channels are both wrong!! only placeholder
        )

        self.image_head = nn.Sequential(
            ConvStack(input_channels=state_input_channels+1, kernel_sizes=(1,5,3), output_channels=(32,32,64)),
            DepthToSpace(block_size=2),
            ConvStack(input_channels=1, kernel_sizes=(3,3,1), output_channels=(64,64,48)), #TODO: input channels are wrong!! only placeholder
            DepthToSpace(block_size=4)
        )

    def forward(self, state, z):
        reward_log_probs = self.reward_head(state)

        #we broadcast z to 1 channel --> TODO: check if this is correct
        broadcasted_z = z.repeat(1, 1, state.shape[2], state.shape[3])
        concatenated = torch.cat((state, broadcasted_z), 1)

        image_log_probs = self.image_head(concatenated)
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
        broadcasted_action = broadcast_action(action=action, num_actions=self.num_actions, broadcast_to_shape=state.shape[2:], use_cuda=self.use_cuda)
        concatenated = torch.cat((state, broadcasted_action), 1)

        mu = self.conv_stack(concatenated)

        sigma = torch.log(1 + torch.exp(mu))    #?? should this be of mu? seems like it in the paper
        return mu, sigma


#Posterior Module for computing mean μ'_z_t and diagonal variance σ'_z_t of the normal distribution q(z_z|s_t-1, a_t-1, o_t).
#   The posterior gets as additional inputs the prior statistics μ_z_t, σ_z_t.
class PosteriorModule(nn.Module):
    def __init__(self, state_input_channels, num_actions):
        super(PosteriorModule, self).__init__()
        #state channels + action broadcast channels + encoded obs always should have 64 channels + 1 channel broadcasted mu + 1 channel broadcasted sigma
        input_channels = state_input_channels + num_actions + 64 + 1 + 1
        self.conv_stack = ConvStack(input_channels=input_channels, kernel_sizes=(1,3,3), output_channels=(32,32,64))

    def forward(self, prev_state, action, encoded_obs, mu, sigma):
        broadcasted_action = broadcast_action(action=action, num_actions=self.num_actions, broadcast_to_shape=prev_state.shape[2:], use_cuda=self.use_cuda)
        broadcasted_mu = mu.repeat(1, 1, prev_state.shape[2], prev_state[3])
        broadcasted_sigma = sigma.repeat(1, 1, prev_state.shape[2], prev_state[3])
        concatenated = torch.cat((prev_state, broadcasted_action, encoded_obs, broadcasted_mu, broadcasted_sigma), 1)

        mu_posterior = self.conv_stack(concatenated)

        sigma_posterior = torch.log(1 + torch.exp(mu_posterior))
        return mu_posterior, sigma_posterior




class InitialStateModule(nn.Module):
    def __init__(self):
        super(InitialStateModule, self).__init__()
        input_channels = 64 + 64 + 64   #encoded states of o_0, o_-1, o_-2 each 64 channels
        self.conv_stack = ConvStack(input_channels=input_channels, kernel_sizes=(1,3,3), output_channels=(64,64,64))

    def forward(self, encoded_now, encoded_prev, encoded_2prev):
        concatenated = torch.cat((encoded_now, encoded_prev, encoded_2prev), 1)
        x = self.conv_stack(concatenated)
        return x