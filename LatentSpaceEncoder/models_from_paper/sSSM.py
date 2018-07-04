import torch.nn as nn
from LatentSpaceEncoder.models_from_paper.model_building_blocks import StateTransition, EncoderModule, InitialStateModule, DecoderModule, PriorModule, PosteriorModule
import torch
from torch.distributions.normal import Normal

class sSSM(nn.Module):
    def __init__(self, observation_input_channels, state_input_channels, num_actions, use_cuda):
        super(sSSM, self).__init__()
        self.encoder = EncoderModule(input_channels=observation_input_channels)
        self.state_transition = StateTransition(state_input_channels=state_input_channels, num_actions=num_actions, use_stochastic=True, use_cuda=use_cuda)
        self.decoder = DecoderModule(state_input_channels=state_input_channels, use_vae=True)
        self.initial_state_module = InitialStateModule()
        self.prior_z = PriorModule(state_input_channels=state_input_channels, num_actions=num_actions, use_cuda=use_cuda)

        self.posterior_z = PosteriorModule(state_input_channels=state_input_channels, num_actions=num_actions, use_cuda=use_cuda)

    def encode(self, observation):
        encoding_t2 = self.encoder(observation[:,-1]) # t0
        encoding_t1 = self.encoder(observation[:,-2]) # t-1
        encoding_t0 = self.encoder(observation[:,-3]) # t-2
        state = self.initial_state_module(encoding_t2, encoding_t1, encoding_t0)

        return state # self.encoder(observation)

    def next_latent_space(self, latent_space, action):
        mu_prior, sigma_prior = self.prior_z(latent_space, action)
        # get latent variable by sampling from prior gaussian
        prior_gaussian = Normal(loc=mu_prior, scale=sigma_prior)
        z_prior = prior_gaussian.sample()
        return self.state_transition(latent_space, action, z_prior), z_prior

    def reward(self, latent_space):
        return self.decoder.reward_head(latent_space)

    def decode(self, latent_space, z_prior):
        return self.decoder(latent_space, z_prior)

    def forward(self, observation, action):
        state = self.encode(observation)
        next_state_prediction, z_prior = self.next_latent_space(state, action)
        image_log_probs, reward_log_probs = self.decoder(next_state_prediction, z_prior)
        return image_log_probs, reward_log_probs


    def forward_multiple(self, observation_initial_context, action_list):
        total_image_log_probs = None
        total_mu_prior = None
        total_sigma_prior = None
        total_mu_posterior = None
        total_sigma_posterior = None

        state = self.encode(observation_initial_context)

        for action in action_list.transpose_(0, 1):
            #compute mean and variance of p(z_t|s_t-1, a_t-1, o_t)
            mu_prior, sigma_prior = self.prior_z(state, action)
            prior_gaussian = Normal(loc=mu_prior, scale=sigma_prior)
            #get latent variable by sampling from prior gaussian
            z_prior = prior_gaussian.sample()

            next_state_prediction = self.state_transition(state, action, z_prior)
            image_log_probs, reward_log_probs = self.decoder(next_state_prediction, z_prior)

            mu_posterior, sigma_posterior = self.posterior_z(prev_state=state, action=action,
                                      encoded_obs=next_state_prediction, mu=mu_prior, sigma=sigma_prior)


            if total_image_log_probs is not None:
                total_image_log_probs = torch.cat((total_image_log_probs, image_log_probs.unsqueeze(1)), dim=1)
                total_mu_prior = torch.cat((total_mu_prior, mu_prior.unsqueeze(1)), dim=1)
                total_sigma_prior = torch.cat((total_sigma_prior, sigma_prior.unsqueeze(1)), dim=1)
                total_mu_posterior = torch.cat((total_mu_posterior, mu_posterior.unsqueeze(1)), dim=1)
                total_sigma_posterior = torch.cat((total_sigma_posterior, sigma_posterior.unsqueeze(1)), dim=1)
            else:
                total_image_log_probs = image_log_probs.unsqueeze(1)
                total_mu_prior = mu_prior.unsqueeze(1)
                total_sigma_prior = sigma_prior.unsqueeze(1)
                total_mu_posterior = mu_posterior.unsqueeze(1)
                total_sigma_posterior = sigma_posterior.unsqueeze(1)

            state = next_state_prediction

        return total_image_log_probs, reward_log_probs, total_mu_prior, total_sigma_prior, total_mu_posterior, total_sigma_posterior