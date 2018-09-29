import torch
from i2a.rollout_encoder import EncoderCNNNetwork, EncoderLSTMNetwork, BasicRolloutEncoder

class ModelBasedNetwork(torch.nn.Module):
    def __init__(self,
                 action_space,
                 obs_shape,
                 imagination_core,
                 number_lstm_cells=256,
                 rollout_steps=5,
                 use_cuda=False):
        super(ModelBasedNetwork, self).__init__()

        self.rollout_steps = rollout_steps
        self.number_lstm_cells = number_lstm_cells
        self.action_space = action_space
        self.use_cuda = use_cuda

        self.imagination_core = imagination_core

        self.encoder_cnn = EncoderCNNNetwork(input_shape=obs_shape)
        # (output size cnn + broadcasted reward)
        self.encoder_lstm = EncoderLSTMNetwork(input_dim=self.encoder_cnn.output_size + self.encoder_cnn.output_dims,
                                               number_lstm_cells=self.number_lstm_cells,
                                               use_cuda=self.use_cuda)

        self.rollout_encoder = BasicRolloutEncoder(self.imagination_core,
                                              self.encoder_cnn,
                                              self.encoder_lstm,
                                              self.rollout_steps,
                                              self.use_cuda)

    def forward(self, input_state):
        # model-based side
        states = input_state.repeat(self.action_space, 1, 1, 1, 1)
        states = states.permute(1, 0, 2, 3, 4).contiguous()
        #batchwise for all rollouts -> each rollout gets a different first action
        actions = torch.arange(self.action_space).long().unsqueeze(1)
        #batchwise for all processes -> repeat the broadcasted actions for each batch
        actions = actions.repeat(input_state.shape[0], 1, 1)
        if self.use_cuda:
            actions = actions.cuda()
        # compute rollout encoder final results

        states_shape = states.shape
        batch_size = states_shape[0] * states_shape[1]
        states = states.view(batch_size, states_shape[2], states_shape[3], states_shape[4])
        actions = actions.view(actions.shape[0] * actions.shape[1], -1)
        self.rollout_encoder.lstm_network.repackage_lstm_hidden_variables(batch_size=batch_size)
        rollout_results = self.rollout_encoder.forward(states, actions)

        # Aggregator: aggregate all lstm outputs
        model_based_result = rollout_results.view(states_shape[0], -1)

        return model_based_result

    def output_size(self):
        return self.number_lstm_cells * self.action_space