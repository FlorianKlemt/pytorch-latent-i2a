import torch.nn as nn
from LatentSpaceEncoder.models_from_paper.model_building_blocks import StateTransition, EncoderModule, DecoderModule
import torch

class dSSM_DET(nn.Module):
    def __init__(self, observation_input_channels, state_input_channels, num_actions, use_cuda):
        super(dSSM_DET, self).__init__()
        self.encoder = EncoderModule(input_channels=observation_input_channels)
        self.state_transition = StateTransition(state_input_channels=state_input_channels, num_actions=num_actions, use_stochastic=False, use_cuda=use_cuda)
        self.decoder = DecoderModule(state_input_channels=state_input_channels, use_vae=False)

        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, observation):
        return self.encoder(observation)

    def next_latent_space(self, latent_space, action):
        return self.state_transition(latent_space, action, None)

    def reward(self, latent_space):
        return self.decoder.reward_head(latent_space)

    def decode(self, latent_space):
        predicted_observation = self.decoder(latent_space)
        return self.sigmoid(predicted_observation)

    def forward(self, observation, action):
        encoding = self.encoder(observation)
        latent_state_prediction = self.state_transition(encoding, action, None) #no latent z for now
        image_log_probs, reward_log_probs = self.decoder(latent_state_prediction, None) #no latent z for now
        image_log_probs = self.sigmoid(image_log_probs)
        return image_log_probs, reward_log_probs

    '''def forward_multiple(self, observation, action_list):
        total_image_log_probs = None
        encoding = self.encoder(observation)

        for action in action_list:
            encoding = self.state_transition(encoding, action, None)  # no latent z for now
            image_log_probs, reward_log_probs = self.decoder(encoding, None)  # no latent z for now
            #image_log_probs = self.log_sigmoid(image_log_probs)
            image_log_probs = torch.clamp(image_log_probs, 0.001, 1.) #???
            image_log_probs = torch.log(image_log_probs)              #???
            if total_image_log_probs is not None:
                total_image_log_probs = torch.cat((total_image_log_probs, image_log_probs))
            else:
                total_image_log_probs = image_log_probs

        #stack of predicted observations
        return total_image_log_probs'''
