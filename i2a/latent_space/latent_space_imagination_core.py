from torch import nn


class LatentSpaceImaginationCore(nn.Module):
    def __init__(self, env_model=None, rollout_policy=None):
        super(LatentSpaceImaginationCore, self).__init__()
        self.env_model = env_model
        self.rollout_policy = rollout_policy

    def forward(self, latent_state, action):
        next_latent_state, z_prior = self.env_model.next_latent_space(latent_state, action)
        reward = self.env_model.reward(next_latent_state)
        return next_latent_state, z_prior, reward

    def encode(self, observation_initial_context):
        latent_space = self.env_model.encode(observation_initial_context)
        return latent_space

    def decode(self, latent_space, z_prior):
        predicted_observation = self.env_model.decode(latent_space, z_prior)
        return predicted_observation

    def sample(self, latent_space):
        value, actor = self.rollout_policy(latent_space)
        action = self.rollout_policy.sample(actor)
        return action
