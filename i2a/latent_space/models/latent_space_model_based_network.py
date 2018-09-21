import torch
from i2a.rollout_encoder import EncoderCNNNetwork, EncoderLSTMNetwork, LatentSpaceRolloutEncoder


class LatentSpaceModelBasedNetwork(torch.nn.Module):
    def __init__(self,
                 action_space,
                 encoding_shape,
                 imagination_core,
                 number_lstm_cells=256,
                 rollout_steps=5,
                 frame_stack=4,
                 use_cuda=False):

        super(LatentSpaceModelBasedNetwork, self).__init__()

        self.rollout_steps = rollout_steps
        self.number_lstm_cells = number_lstm_cells
        self.action_space = action_space
        self.use_cuda = use_cuda
        self.frame_stack = frame_stack

        self.imagination_core = imagination_core

        self.encoder_cnn = EncoderCNNNetwork(input_shape=encoding_shape)
        # (output size cnn + broadcasted reward)
        self.encoder_lstm = EncoderLSTMNetwork(input_dim=self.encoder_cnn.output_size + self.encoder_cnn.output_dims,
                                                          number_lstm_cells=self.number_lstm_cells,
                                                          use_cuda=self.use_cuda)

        self.rollout_encoder = LatentSpaceRolloutEncoder(self.imagination_core,
                                              self.encoder_cnn,
                                              self.encoder_lstm,
                                              self.rollout_steps,
                                              self.use_cuda)

    def forward(self, observation_initial_context):
        # model-based side
        states_shape = observation_initial_context.shape
        observation_initial_context = observation_initial_context.view(states_shape[0], self.frame_stack, -1, states_shape[2], states_shape[3])
        latent_space = self.rollout_encoder.imagination_core.encode(observation_initial_context)

        latent_space = latent_space.repeat(self.action_space, 1, 1, 1)
        #batchwise for all rollouts -> each rollout gets a different first action
        #the unsqueeze is needed because a inplace scatter deep inside the rollout encoder needs this dimensionality
        actions = torch.arange(self.action_space).long().unsqueeze(1)
        #batchwise for all processes -> repeat the broadcasted actions for each batch
        actions = actions.repeat(states_shape[0], 1)
        if self.use_cuda:
            actions = actions.cuda()
        # compute rollout encoder final results

        self.rollout_encoder.lstm_network.repackage_lstm_hidden_variables(batch_size=latent_space.shape[0])
        rollout_results = self.rollout_encoder.forward(latent_space, actions)

        # Aggregator: aggregate all lstm outputs
        model_based_result = rollout_results.view(states_shape[0], -1)

        return model_based_result

    def output_size(self):
        return self.number_lstm_cells * self.action_space