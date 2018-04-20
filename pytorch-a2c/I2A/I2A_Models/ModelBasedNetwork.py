import torch
from I2A.RolloutEncoder import EncoderCNNNetwork, EncoderLSTMNetwork, RolloutEncoder
from I2A.ImaginationCore import MiniPacmanImaginationCore

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

        self.imagination_core = MiniPacmanImaginationCore(num_inputs=input_channels, use_cuda=self.use_cuda, require_grad=True)  #here policy grads are required

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
        model_based_results = []

        # for each process
        for i in range(input_state.data.shape[0]):
            self.repackage_lstm_hidden_variables()
            state = input_state[i].unsqueeze(0)
            # compute rollout encoder final results
            rollout_results = []
            for rollout_encoder in self.rollout_encoder_list:
                rollout_results.append(rollout_encoder.forward(state))

            # Aggregator: aggregate all lstm outputs
            process_result = torch.cat(rollout_results, 1)
            model_based_results.append(process_result)

        # Aggregator: aggregate all lstm outputs
        model_based_result = torch.cat(model_based_results, 0)

        return model_based_result

    def repackage_lstm_hidden_variables(self):
        for rollout_encoder in self.rollout_encoder_list:
            rollout_encoder.repackage_lstm_hidden_variables()
