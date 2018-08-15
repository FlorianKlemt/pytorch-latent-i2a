import torch

# the copy model returns the identity,
# this is its own class so we dont have to change the code to use the copymodel
class CopyEnvModel(torch.nn.Module):
    def __init__(self):
        super(CopyEnvModel, self).__init__()
    def forward(self, input_frame, input_action):
        return input_frame, torch.zeros(input_frame.shape[0]).cuda()