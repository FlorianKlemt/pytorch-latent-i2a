from __future__ import division
import numpy as np
import torch
import math
from functools import reduce
import json
import logging

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def get_linear_dims_after_conv(conv_list, input_dims):
    conv_output_dims = get_conv_output_dims(conv_list, input_dims)
    return conv_list[-1].out_channels * reduce(lambda x, y: x * y, conv_output_dims)

def get_conv_output_dims(conv_list, input_dim):
    intermediate_dims = input_dim
    for conv_layer in conv_list:
        intermediate_dims = get_single_conv_output_dims(conv_layer, intermediate_dims)
    return intermediate_dims

#this method only works for 2 input dimensions
def get_single_conv_output_dims(conv_layer, input_dims):
    assert len(input_dims)==2
    out_dim_1 = ((input_dims[0] - conv_layer.kernel_size[0] + 2 * conv_layer.padding[0]) / conv_layer.stride[0]) + 1
    out_dim_2 = ((input_dims[1] - conv_layer.kernel_size[1] + 2 * conv_layer.padding[1]) / conv_layer.stride[1]) + 1
    return math.floor(out_dim_1), math.floor(out_dim_2)

import torch.nn as nn
def get_possible_sizes(conv_list, input_dims):
    out_list = []
    intermediate_dims = input_dims
    for conv_layer in conv_list:
        k_1 = conv_layer.kernel_size[0] - 1
        out_dim_1 = 0.5
        while not out_dim_1.is_integer():
            k_1 += 1
            out_dim_1 = ((intermediate_dims[0] - k_1) / conv_layer.stride[0]) + 1
        k_2 = conv_layer.kernel_size[1] - 1
        out_dim_2 = 0.5
        while not out_dim_2.is_integer():
            k_2 += 1
            out_dim_2 = ((intermediate_dims[1] - k_2) / conv_layer.stride[1]) + 1
        intermediate_dims = (out_dim_1,out_dim_2)
        out_list.append(nn.Conv2d(in_channels=conv_layer.in_channels, out_channels=conv_layer.out_channels,
                                  kernel_size=(k_1,k_2), stride=conv_layer.stride, padding=conv_layer.padding))
    return out_list