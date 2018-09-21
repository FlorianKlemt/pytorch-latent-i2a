import torch
import torch.nn as nn
import torch.nn.functional as F
from environment_model.latent_space.models_from_paper.model_building_blocks import ResConv, PoolAndInject, broadcast_action


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