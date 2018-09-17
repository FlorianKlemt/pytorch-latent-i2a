import torch.nn as nn
from environment_model.latent_space.models_from_paper.model_building_blocks import PosteriorModule, PriorModule, StateTransition, EncoderModule, DecoderModule, InitialStateModule
import torch
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import math

class SSM(nn.Module):
    def __init__(self,
                 model_type,
                 observation_input_channels,
                 state_input_channels,
                 num_actions,
                 use_cuda,
                 reward_prediction_bits = 8):
        super(SSM, self).__init__()
        self.use_cuda = use_cuda
        self.encoder = EncoderModule(input_channels=observation_input_channels)
        self.initial_state_module = InitialStateModule()
        self.state_transition = StateTransition(state_input_channels=state_input_channels,
                                                num_actions=num_actions, use_stochastic=False, use_cuda=use_cuda)
        self.decoder = DecoderModule(state_input_channels=state_input_channels,
                                     use_vae=False,
                                     reward_prediction_bits=reward_prediction_bits)
        self.prior_z = PriorModule(state_input_channels=state_input_channels,
                                   num_actions=num_actions,
                                   use_cuda=use_cuda)

        if model_type == "dSSM_DET":
            self.get_prior = _get_prior_dSSM_DET
            self._forward_multiple = self._forward_multiple_DSSM
        elif model_type == "dSSM_VAE":
            self.get_prior = _get_prior_dSSM_VAE
            self._forward_multiple = self._forward_multiple_Z
            self.posterior_z = PosteriorModule(state_input_channels=state_input_channels,
                                               num_actions=num_actions,
                                               use_cuda=use_cuda)
        elif model_type == "sSSM":
            self.get_prior = _get_prior_sSSM
            self._forward_multiple = self._forward_multiple_Z
            self.posterior_z = PosteriorModule(state_input_channels=state_input_channels,
                                               num_actions=num_actions,
                                               use_cuda=use_cuda)
        else:
            assert False #lol

    def encode(self, observation):
        encoding_t2 = self.encoder(observation[:,-1]) # t0
        encoding_t1 = self.encoder(observation[:,-2]) # t-1
        encoding_t0 = self.encoder(observation[:,-3]) # t-2
        state = self.initial_state_module(encoding_t2, encoding_t1, encoding_t0)
        return state

    def reward(self, latent_space):
        reward_logits = self.decoder.reward_head(latent_space)
        return self.get_numerical_reward(reward_logits)

    # convert predicted to number
    def get_numerical_reward(self, reward_logits):
        reward_bernoulli = Bernoulli(logits=reward_logits)
        sampled_r = reward_bernoulli.sample()

        r_out = torch.FloatTensor(sampled_r.shape[0], 1).fill_(0)
        if self.use_cuda:
            r_out = r_out.cuda()
        for i in range(sampled_r.shape[0]):
            r = 0
            if sampled_r[i, 0] == 1:
                r_out[i] = 0
            else:
                for k in range(2, sampled_r.shape[1]):
                    r += int(math.pow(2, sampled_r.shape[1] - 1 - k)) if sampled_r[i, k] == 1 else 0
                if sampled_r[i, 1] == 1:
                    r = -r
                r_out[i] = r
        return r_out

        '''r_out = torch.cuda.FloatTensor(sampled_r.shape[0], sampled_r.shape[1]).fill_(0)
        for i in range(sampled_r.shape[0]):
            for j in range(sampled_r.shape[1]):
                r = 0
                if sampled_r[i,j,0] == 1:
                    r_out[i,j] = 0
                else:
                    for k in range(2,sampled_r.shape[2]):
                        r += int(math.pow(2,sampled_r.shape[2]-1-k)) if sampled_r[i,j,k]==1 else 0
                    if sampled_r[i,j,1] == 1:
                        r = -r
                    r_out[i,j] = r
        '''

    '''def get_numerical_reward_2(self):
        reward_bernoulli = Bernoulli(logits=reward_logits)
        sampled_r = reward_bernoulli.sample()

        r_out = torch.cuda.FloatTensor(sampled_r.shape[0], sampled_r.shape[1]).fill_(0)
        r_out[sampled_r[:, :, 0] == 1] = 0
        for i in range(2, 6):
            r_out[sampled_r[:, :, i] == 1] = int(math.pow(2,sampled_r.shape[2]-1-k))
  
        r_out[sampled_r[:, :, 1] == 1] *= -1'''


    def decode(self, latent_space, z_prior):
        return self.decoder(latent_space, z_prior)

    def forward(self, observation_initial_context, action):
        encoding = self.encode(observation_initial_context)
        latent_state_prediction, z_prior = self.next_latent_space(encoding, action)
        image_log_probs, reward_log_probs = self.decode(latent_state_prediction, z_prior) #no latent z for now
        return image_log_probs, reward_log_probs


    def next_latent_space(self, latent_space, action):
        mu_prior, sigma_prior = self.prior_z(latent_space, action)
        state_transition_z, decoder_z = self.get_prior(mu_prior, sigma_prior)
        return self.state_transition(latent_space, action, state_transition_z), decoder_z

    def forward_multiple(self, observation_initial_context, action_list):
        return self._forward_multiple(observation_initial_context, action_list)


    def _forward_multiple_DSSM(self, observation_initial_context, action_list):
        total_image_log_probs = None
        total_reward_log_probs = None
        state = self.encode(observation_initial_context)
        # iterate over T actions, but pass action t for all batches simultaneously
        for action in action_list.transpose_(0, 1):
            state, decoder_z = self.next_latent_space(state, action)
            image_log_probs, reward_log_probs = self.decode(state, decoder_z)


            if total_image_log_probs is not None:
                total_image_log_probs = torch.cat((total_image_log_probs, image_log_probs.unsqueeze(1)), dim=1)
                total_reward_log_probs = torch.cat((total_reward_log_probs, reward_log_probs.unsqueeze(1)), dim=1)
            else:
                total_image_log_probs = image_log_probs.unsqueeze(1)
                total_reward_log_probs = reward_log_probs.unsqueeze(1)
        return total_image_log_probs, total_reward_log_probs


    def _forward_multiple_Z(self, observation_initial_context, action_list):
        total_mu_prior = None
        total_sigma_prior = None
        total_mu_posterior = None
        total_sigma_posterior = None

        total_image_log_probs = None
        total_reward_log_probs = None

        state = self.encode(observation_initial_context)
        # iterate over T actions, but pass action t for all batches simultaneously
        for action in action_list.transpose_(0, 1):
            # get nect latent space
            mu_prior, sigma_prior = self.prior_z(state, action)
            state_transition_z, decoder_z = self.get_prior(mu_prior, sigma_prior)
            next_state_prediction = self.state_transition(state, action, state_transition_z)

            image_log_probs, reward_log_probs = self.decode(next_state_prediction, decoder_z)

            mu_posterior, sigma_posterior = self.posterior_z(prev_state=state,
                                                             action=action,
                                                             encoded_obs=next_state_prediction,
                                                             mu=mu_prior, sigma=sigma_prior)

            if total_mu_posterior is not None:
                total_mu_prior = torch.cat((total_mu_prior, mu_prior.unsqueeze(1)), dim=1)
                total_sigma_prior = torch.cat((total_sigma_prior, sigma_prior.unsqueeze(1)), dim=1)
                total_mu_posterior = torch.cat((total_mu_posterior, mu_posterior.unsqueeze(1)), dim=1)
                total_sigma_posterior = torch.cat((total_sigma_posterior, sigma_posterior.unsqueeze(1)), dim=1)
            else:
                total_mu_prior = mu_prior.unsqueeze(1)
                total_sigma_prior = sigma_prior.unsqueeze(1)
                total_mu_posterior = mu_posterior.unsqueeze(1)
                total_sigma_posterior = sigma_posterior.unsqueeze(1)

            if total_image_log_probs is not None:
                total_image_log_probs = torch.cat((total_image_log_probs, image_log_probs.unsqueeze(1)), dim=1)
                total_reward_log_probs = torch.cat((total_reward_log_probs, reward_log_probs.unsqueeze(1)), dim=1)
            else:
                total_image_log_probs = image_log_probs.unsqueeze(1)
                total_reward_log_probs = reward_log_probs.unsqueeze(1)

            state = next_state_prediction

        latent_variables = (total_mu_prior, total_sigma_prior, total_mu_posterior, total_sigma_posterior)
        return total_image_log_probs, total_reward_log_probs, latent_variables


def _get_prior_dSSM_DET(mu_prior, sigma_prior):
    return mu_prior, mu_prior

def _get_prior_dSSM_VAE(mu_prior, sigma_prior):
    prior_gaussian = Normal(loc=mu_prior, scale=sigma_prior)
    z_prior = prior_gaussian.sample()
    # state transition gets the mean of the prior distribution instead of a sample
    # in dSSM_VAE, for gaussian the mean is mu_prior
    return mu_prior, z_prior

def _get_prior_sSSM(mu_prior, sigma_prior):
    prior_gaussian = Normal(loc=mu_prior, scale=sigma_prior)
    z_prior = prior_gaussian.sample()
    return z_prior, z_prior
