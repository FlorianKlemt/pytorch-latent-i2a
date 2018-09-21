import torch

class Flatten(torch.nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)