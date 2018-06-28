import torch.nn as nn
from LatentSpaceEncoder.models_from_paper.model_building_blocks import StateTransition, EncoderModule, DecoderModule, PriorModule, PosteriorModule
import torch

class sSSM(nn.Module):
    def __init__(self, observation_input_channels, state_input_channels, num_actions, use_cuda):
        super(sSSM, self).__init__()
        self.encoder = EncoderModule(input_channels=observation_input_channels)
        self.state_transition = StateTransition(state_input_channels=state_input_channels, num_actions=num_actions, use_stochastic=True, use_cuda=use_cuda)
        self.decoder = DecoderModule(state_input_channels=state_input_channels, use_vae=True)

        self.prior_z = PriorModule(state_input_channels=state_input_channels, num_actions=num_actions, use_cuda=use_cuda)

        self.posterior_z = PosteriorModule(state_input_channels=state_input_channels, num_actions=num_actions, use_cuda=use_cuda)

    def forward(self, observation, action):
        #TODO: second forward function which takes state and action
        state = self.encoder(observation)

        z = self.prior_z(state, action)

        next_state_prediction = self.state_transition(state, action, z)
        image_log_probs, reward_log_probs = self.decoder(next_state_prediction, z)

        return image_log_probs, reward_log_probs


    def forward_multiple(self, observation, action_list):
        total_image_log_probs = None
        total_z_mu_prior = None
        total_z_sigma_prior = None
        total_z_mu_posterior = None
        total_z_sigma_posterior = None
        state = self.encoder(observation[:,0])  #there is a batch for an initial context, even though it is not used here

        for action in action_list.transpose_(0, 1):
            z_prior = self.prior_z(state, action)

            next_state_prediction = self.state_transition(state, action, z_prior)
            image_log_probs, reward_log_probs = self.decoder(next_state_prediction, z_prior)

            prior_z_mu, prior_z_sigma = z_prior

            z_posterior = self.posterior_z(prev_state=state, action=action,
                                      encoded_obs=next_state_prediction, mu=prior_z_mu, sigma=prior_z_sigma)
            posterior_z_mu, posterior_z_sigma = z_posterior

            if total_image_log_probs is not None:
                total_image_log_probs = torch.cat((total_image_log_probs, image_log_probs.unsqueeze(1)), dim=1)
                total_z_mu_prior = torch.cat((total_z_mu_prior, prior_z_mu.unsqueeze(1)), dim=1)
                total_z_sigma_prior = torch.cat((total_z_sigma_prior, prior_z_sigma.unsqueeze(1)), dim=1)
                total_z_mu_posterior = torch.cat((total_z_mu_posterior, posterior_z_mu.unsqueeze(1)), dim=1)
                total_z_sigma_posterior = torch.cat((total_z_sigma_posterior, posterior_z_sigma.unsqueeze(1)), dim=1)
            else:
                total_image_log_probs = image_log_probs.unsqueeze(1)
                total_z_mu_prior = prior_z_mu.unsqueeze(1)
                total_z_sigma_prior = prior_z_sigma.unsqueeze(1)
                total_z_mu_posterior = posterior_z_mu.unsqueeze(1)
                total_z_sigma_posterior = posterior_z_sigma.unsqueeze(1)

            state = next_state_prediction

        return total_image_log_probs, reward_log_probs, total_z_mu_prior, total_z_sigma_prior, total_z_mu_posterior, total_z_sigma_posterior