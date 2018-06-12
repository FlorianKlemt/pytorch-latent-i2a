import torch
from torch import nn


class ImaginationCore(nn.Module):
    def __init__(self, env_model=None, rollout_policy=None, grey_scale=False, frame_stack=4):
        super(ImaginationCore, self).__init__()
        self.env_model = env_model
        self.rollout_policy = rollout_policy
        self.grey_scale = grey_scale
        self.frame_stack = frame_stack

    def forward(self, state, action):
        if self.frame_stack > 1:
            # TODO: check this (hopefully the last 3 elements are RGB of the last frame)
            last_frame = state[:,-1:] if self.grey_scale else state[:,-3:]
        else:
            last_frame = state
        next_state, reward = self.env_model(last_frame, action)
        if self.frame_stack > 1:
            x = torch.cat((state, next_state), 1)
            next_state = x[:, 1:] if self.grey_scale else x[:, 3:]
        return next_state, reward

    def sample(self, input_state):
        value, action, action_log_probs, states = self.rollout_policy.act(input_state, None, None)
        return action
