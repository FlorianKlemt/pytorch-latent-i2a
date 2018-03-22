from I2A.RolloutEncoder import *
from I2A.ImaginationCore import PongImaginationCore

class ModelBasedNetwork(torch.nn.Module):
    """
    This class implements the model free A3CLstm architecture implemented, the code is based on
    https://github.com/dgriff777/rl_a3c_pytorch. We also used the pretrained models if available.
    """

    def __init__(self,
                 number_actions,
                 input_channels,
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
        self.input_channels = input_channels
        self.number_lstm_cells = number_lstm_cells
        self.number_actions = number_actions
        self.use_cuda = use_cuda

        self.imagination_core = PongImaginationCore(self.use_cuda)

        self.encoder_cnn = EncoderCNNNetwork(self.input_channels)
        self.encoder_lstm = EncoderLSTMNetwork(self.number_lstm_cells, use_cuda=self.use_cuda)

        self.rollout_encoder_list = []
        for i in range(self.number_actions):
            rollout = RolloutEncoder(self.imagination_core,
                                     self.encoder_cnn,
                                     self.encoder_lstm,
                                     self.rollout_steps,
                                     i,
                                     self.use_cuda)
            self.rollout_encoder_list.append(rollout)

    def forward(self, input_state):
        # model-based side
        # compute rollout encoder final results
        rollout_results = []
        for rollout_encoder in self.rollout_encoder_list:
            rollout_results.append(rollout_encoder.forward(input_state))

        # Aggregator: aggregate all lstm outputs
        model_based_result = torch.cat(rollout_results, 1)

        return model_based_result

    def repackage_lstm_hidden_variables(self):
        for rollout_encoder in self.rollout_encoder_list:
            rollout_encoder.repackage_lstm_hidden_variables()
