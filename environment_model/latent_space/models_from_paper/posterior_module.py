import torch
import torch.nn as nn
from environment_model.latent_space.models_from_paper.model_building_blocks import ConvStack, broadcast_action


# Posterior Module for computing mean μ'_z_t and diagonal variance σ'_z_t of the normal distribution q(z_z|s_t-1, a_t-1, o_t).
# The posterior gets as additional inputs the prior statistics μ_z_t, σ_z_t.
class PosteriorModule(nn.Module):
    def __init__(self, state_input_channels, num_actions, use_cuda):
        super(PosteriorModule, self).__init__()
        self.use_cuda = use_cuda
        self.num_actions = num_actions
        # Explanation: input channels = state channels + action broadcast channels
        #                               + 64 channels encoded observation + 64 channel broadcasted mu + 64 channel broadcasted sigma
        input_channels = state_input_channels + num_actions + 64 + 64 + 64
        self.conv_stack = ConvStack(input_channels=input_channels, kernel_sizes=(1,3,3), output_channels=(32,32,64))

    def forward(self, prev_state, action, encoded_obs, mu, sigma):
        broadcasted_action = broadcast_action(action=action, num_actions=self.num_actions, broadcast_to_shape=prev_state.shape[2:], use_cuda=self.use_cuda)
        concatenated = torch.cat((prev_state, broadcasted_action, encoded_obs, mu, sigma), 1)

        mu_posterior = self.conv_stack(concatenated)

        sigma_posterior = torch.log(1 + torch.exp(mu_posterior))
        return mu_posterior, sigma_posterior


