import torch.nn as nn
from LatentSpaceEncoder.models_from_paper.model_building_blocks import StateTransition, EncoderModule, DecoderModule, InitialStateModule
import torch

class dSSM_DET(nn.Module):
    def __init__(self, observation_input_channels, state_input_channels, num_actions, use_cuda):
        super(dSSM_DET, self).__init__()
        self.encoder = EncoderModule(input_channels=observation_input_channels)
        self.initial_state_module = InitialStateModule()
        self.state_transition = StateTransition(state_input_channels=state_input_channels, num_actions=num_actions, use_stochastic=False, use_cuda=use_cuda)
        self.decoder = DecoderModule(state_input_channels=state_input_channels, use_vae=False)

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
        #TODO: this needs to be done with the 3 last frames for testing
        encoding = self.initial_state_module(encoding, encoding, encoding)
        latent_state_prediction = self.state_transition(encoding, action, None) #no latent z for now
        image_log_probs, reward_log_probs = self.decoder(latent_state_prediction, None) #no latent z for now
        return image_log_probs, reward_log_probs

    def forward_multiple(self, observation_initial_context, action_list):
        total_image_log_probs = None

        #get initial context
        t0_encoding = self.encoder(observation_initial_context[:,2])
        one_before_encoding = self.encoder(observation_initial_context[:,1])
        two_before_encoding = self.encoder(observation_initial_context[:,0])
        state = self.initial_state_module(t0_encoding, one_before_encoding, two_before_encoding)

        #iterate over T actions, but pass action t for all batches simultaneously
        for action in action_list.transpose_(0, 1):
            state = self.state_transition(state, action, None)  # no latent z for now
            image_log_probs, reward_log_probs = self.decoder(state, None)  # no latent z for now

            #image_log_probs are unsqueezed at 1 to create a stack dimension between batch_dimension(0) and channel_dimension(1)
            if total_image_log_probs is not None:
                total_image_log_probs = torch.cat((total_image_log_probs, image_log_probs.unsqueeze(1)), dim=1)
            else:
                total_image_log_probs = image_log_probs.unsqueeze(1)

        #stack of predicted observations
        return total_image_log_probs
