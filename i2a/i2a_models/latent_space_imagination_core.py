import torch
from torch import nn


class LatentSpaceImaginationCore(nn.Module):
    def __init__(self, env_model=None, rollout_policy=None, grey_scale=False, frame_stack=4):
        super(LatentSpaceImaginationCore, self).__init__()
        self.env_model = env_model
        self.rollout_policy = rollout_policy

    def forward(self, latent_state, action):
        next_latent_state, z_prior = self.env_model.next_latent_space(latent_state, action)
        reward = self.env_model.reward(next_latent_state)
        return next_latent_state, z_prior, reward

    def encode(self, state):
        latent_space = self.env_model.encode(state)
        return latent_space

    def decode(self, state, z_prior):
        latent_space = self.env_model.decode(state, z_prior)
        return latent_space

    def sample(self, input_state):
        value, actor = self.rollout_policy(input_state)
        action = self.rollout_policy.sample(actor)
        return action
