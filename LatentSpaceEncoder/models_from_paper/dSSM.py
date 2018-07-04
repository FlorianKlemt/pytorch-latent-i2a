import torch.nn as nn
from LatentSpaceEncoder.models_from_paper.model_building_blocks import PriorModule, StateTransition, EncoderModule, DecoderModule, InitialStateModule
import torch

class dSSM_DET(nn.Module):
    def __init__(self, observation_input_channels, state_input_channels, num_actions, use_cuda):
        super(dSSM_DET, self).__init__()
        self.encoder = EncoderModule(input_channels=observation_input_channels)
        self.initial_state_module = InitialStateModule()
        self.state_transition = StateTransition(state_input_channels=state_input_channels, num_actions=num_actions, use_stochastic=False, use_cuda=use_cuda)
        self.decoder = DecoderModule(state_input_channels=state_input_channels, use_vae=False)

        self.prior_z = PriorModule(state_input_channels=state_input_channels, num_actions=num_actions,
                                   use_cuda=use_cuda)

    def encode(self, observation):
        encoding_t2 = self.encoder(observation[:,-1]) # t0
        encoding_t1 = self.encoder(observation[:,-2]) # t-1
        encoding_t0 = self.encoder(observation[:,-3]) # t-2
        state = self.initial_state_module(encoding_t2, encoding_t1, encoding_t0)
        return state

    def next_latent_space(self, latent_space, action):
        mu_prior, sigma_prior = self.prior_z(latent_space, action)
        # state transition gets the mean of the prior distribution instead of a sample
        # in dSSM_DET, for gaussian the mean is mu_prior
        return self.state_transition(latent_space, action, mu_prior), mu_prior

    def reward(self, latent_space):
        return self.decoder.reward_head(latent_space)

    def decode(self, latent_space, z_prior):
        return self.decoder(latent_space, z_prior)

    def forward(self, observation_initial_context, action):
        encoding = self.encode(observation_initial_context)
        latent_state_prediction, z_prior = self.next_latent_space(encoding, action)
        image_log_probs, reward_log_probs = self.decode(latent_state_prediction, z_prior) #no latent z for now
        return image_log_probs, reward_log_probs

    def forward_multiple(self, observation_initial_context, action_list):
        total_image_log_probs = None

        state = self.encode(observation_initial_context)

        #iterate over T actions, but pass action t for all batches simultaneously
        for action in action_list.transpose_(0, 1):
            state, z_prior = self.next_latent_space(state, action)
            image_log_probs, reward_log_probs = self.decode(state, z_prior)  # no latent z for now

            #image_log_probs are unsqueezed at 1 to create a stack dimension between batch_dimension(0) and channel_dimension(1)
            if total_image_log_probs is not None:
                total_image_log_probs = torch.cat((total_image_log_probs, image_log_probs.unsqueeze(1)), dim=1)
            else:
                total_image_log_probs = image_log_probs.unsqueeze(1)

        #stack of predicted observations
        return total_image_log_probs
