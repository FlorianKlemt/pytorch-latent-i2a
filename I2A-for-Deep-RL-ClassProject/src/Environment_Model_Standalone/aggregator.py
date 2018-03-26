import torch
from torch import nn


class PathAggregator(nn.Module):
    """
        This class is used to aggregate the model free path and imagination path.

        Example Usage:
        > import torch
        > from aggregator import Aggregator
        > x_1 = torch.Tensor([[0,1,2], [1,1,1]]) # Dummy output model free
        > x_2 = torch.Tensor([[1,2,3], [2,2,2]]) # Dummy output imagination
        > agg = PathAggregator()
        > agg(x_1, x_2)
        ...  0  1  2  1  2  3
        ... 1  1  1  2  2  2
        ... [torch.FloatTensor of size 2x6]
    """

    def __init__(self):
        super(PathAggregator, self).__init__()

    def forward(self, x_1, x_2):
        if x_1.size()[0] != x_2.size()[0]:
            raise IndexError("Expected two same sized objects." +
                             "x_1.size()=={} and x_2.size()=={}" \
                             .format(x_1.size(), x_2.size()))

        x = torch.cat([x_1, x_2], dim=1)
        return x


class RolloutAggregator(nn.Module):
    """
        This class is used to aggregate the outputs of the different rollout encodings.

        Example Usage:
        > import torch
        > from aggregator import Aggregator
        > x_1 = torch.Tensor([[0,1,2], [1,1,1]]) # Dummy output model free
        > x_2 = torch.Tensor([[1,2,3], [2,2,2]]) # Dummy output imagination
        > agg = RolloutAggregator()
        > agg(x_1, x_2)
        ...  0  1  2  1  2  3
        ... 1  1  1  2  2  2
        ... [torch.FloatTensor of size 2x6]
    """

    def __init__(self):
        super(RolloutAggregator, self).__init__()

    def forward(self, *x_s):
        input_size = x_s[0].size()
        for x in list(x_s):
            if x.size() != input_size:
                raise IndexError("Expected same sized objects." +
                                 "Expected: {} but was {}" \
                                 .format(input_size, x.size()))

        x = torch.cat(list(x_s), dim=1)
        return x