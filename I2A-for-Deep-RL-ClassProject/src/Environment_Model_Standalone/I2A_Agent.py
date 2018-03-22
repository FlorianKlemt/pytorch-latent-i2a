import torch
from torch.autograd import Variable

from I2A.RolloutEncoder import *
from I2A.models.ModelFreeNetwork import ModelFreeNetwork

class I2A():
    def __init__(self, imagination_core, encoder_network):
        #todo get this from env
        self.nr_actions = 5
        self.input_channels = 1
        self.rollout_steps = 5

        self.output_policy_network = OutputPolicyNetwork()

        #model-free side
        self.model_free_network = ModelFreeNetwork()

        #model-based side
        #TODO: implement Imagination Core, should consist of Policy Net and EnvironmentModel, should implement a forward pass
        self.imagination_core = ImaginationCore(None,None)
        self.encoder_cnn = EncoderCNNNetwork(self.input_channels)
        self.encoder_lstm = EncoderLSTMNetwork()
        self.rollout_encoder_list = []
        for i in range(self.nr_actions):
            self.rollout_encoder_list.append(
                RolloutEncoder(self.imagination_core, self.encoder_cnn, self.encoder_lstm, self.rollout_steps, i)
            )


    def forward(self, input_state):
        #model-free side
        model_free_result = self.model_free_network.forward(input_state)

        #model-based side
        #compute rollout encoder final results
        rollout_results = []
        for rollout_encoder in self.rollout_encoder_list:
            rollout_results.append(rollout_encoder.forward(input_state))

        #Aggregator: aggregate all lstm outputs
        model_based_result = torch.cat(rollout_results, 0) #TODO: check if 0 is the correct dim
        #print(len(model_based_result)) - 1280

        #aggregate model-free and model-based side
        aggregated_results = torch.cat((model_free_result,model_based_result), 0)

        return self.output_policy_network.forward(aggregated_results)


#i2a = I2A(None,None)
#input_state = Variable(torch.from_numpy(np.ones(shape=(1,1,80,80)))).type(FloatTensor)
#i2a.forward(input_state)