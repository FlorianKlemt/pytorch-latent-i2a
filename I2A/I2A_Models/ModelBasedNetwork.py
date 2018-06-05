import torch
from I2A.RolloutEncoder import EncoderCNNNetwork, EncoderLSTMNetwork, RolloutEncoder
import numpy as np

class ModelBasedNetwork(torch.nn.Module):
    """
    This class implements the model free A3CLstm architecture implemented, the code is based on
    https://github.com/dgriff777/rl_a3c_pytorch. We also used the pretrained models if available.
    """

    def __init__(self,
                 number_actions,
                 obs_shape,
                 imagination_core,
                 number_lstm_cells=256,
                 rollout_steps=5,
                 use_cuda=False):
        """
        The constructor need two integers, where the num_inputs describes the number of
        input channels and num_outputs is corresponding with the number of actions available in
        the specific game.
        :param num_inputs:
        :param num_outputs:
        """
        super(ModelBasedNetwork, self).__init__()

        self.rollout_steps = rollout_steps
        self.number_lstm_cells = number_lstm_cells
        self.number_actions = number_actions
        self.use_cuda = use_cuda

        self.imagination_core = imagination_core

        self.encoder_cnn = EncoderCNNNetwork(obs_shape=obs_shape)
        # (output size cnn + broadcasted reward)
        self.encoder_lstm = EncoderLSTMNetwork(input_dim=self.encoder_cnn.output_size + self.encoder_cnn.output_dims,
                                               number_lstm_cells=self.number_lstm_cells,
                                               use_cuda=self.use_cuda)

        self.rollout_encoder = RolloutEncoder(self.imagination_core,
                                              self.encoder_cnn,
                                              self.encoder_lstm,
                                              self.rollout_steps,
                                              self.use_cuda)

    def forward(self, input_state):
        # model-based side
        states = input_state.repeat(self.number_actions, 1, 1, 1, 1)
        states = states.permute(1, 0, 2, 3, 4).contiguous()
        actions = torch.arange(self.number_actions).long().unsqueeze(1)
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
