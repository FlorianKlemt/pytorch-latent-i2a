import torch
import torch.nn as nn
from environment_model.latent_space.models_from_paper.model_building_blocks import ConvStack, broadcast_action

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