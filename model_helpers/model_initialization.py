import torch.nn as nn
import torch.nn.init as init
import math

def xavier_weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

def xavier_weights_init_relu(m):
    if isinstance(m, nn.Conv2d):
        xavier_weights_init(m)
        m.weight.data.mul_(math.sqrt(2))