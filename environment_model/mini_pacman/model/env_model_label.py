import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from environment_model.mini_pacman.model.basic_blocks import BasicBlock
from model_helpers.flatten import Flatten
from model_helpers.model_initialization import xavier_weights_init


class MiniPacmanEnvModelClassLabels(torch.nn.Module):
    def __init__(self, obs_shape, num_actions, reward_bins, use_cuda):
        super(MiniPacmanEnvModelClassLabels, self).__init__()
        self.num_actions = num_actions

        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        self.reward_bins = self.FloatTensor(reward_bins)

        input_channels = obs_shape[0]
        W=obs_shape[1]
        H=obs_shape[2]
        self.conv1 = nn.Conv2d(input_channels+self.num_actions, 64, kernel_size=1, stride=1, padding=0)    #input size is channels of input frame +1 for the broadcasted action
        self.basic_block1 = BasicBlock(num_inputs=64,n1=16,n2=32,n3=64,W=W,H=H,use_cuda=use_cuda)
        self.basic_block2 = BasicBlock(num_inputs=64, n1=16, n2=32, n3=64, W=W, H=H,use_cuda=use_cuda)
        self.reward_head = nn.Sequential(OrderedDict([
          ('reward_conv1', nn.Conv2d(64, 64, kernel_size=1)),  #input size is n3 of basic-block2
          ('reward_relu1', nn.ReLU()),
          ('reward_conv2', nn.Conv2d(64, 64, kernel_size=1)),
          ('reward_relu2', nn.ReLU()),
          ('flatten',      Flatten()),
          ('reward_fc',    nn.Linear(64*W*H, 5)),
          ('softmax', nn.Softmax(dim=1))
        ]))

        self.img_head = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, input_channels, kernel_size=1))
        ]))

        self.apply(xavier_weights_init)

        self.train()

        self.rgb_to_class = MiniPacmanRGBToClassConverter(use_cuda=use_cuda)

    def forward(self,input_frame,input_action):
        input_frame = self.rgb_to_class.minipacman_rgb_to_class(input_frame)

        image_out, reward_out = self.forward_class(input_frame, input_action)

        image_out = self.rgb_to_class.minipacman_class_to_rgb(image_out)

        return image_out,reward_out

    def forward_class(self,input_frame,input_action):
        one_hot = torch.zeros(input_action.shape[0], self.num_actions).type(self.FloatTensor)
        # make one hot vector
        one_hot.scatter_(1, input_action, 1)
        # broadcast action
        one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)
        broadcasted_action = one_hot.repeat(1, 1, input_frame.shape[2], input_frame.shape[3])

        # concatinate observation and broadcasted action
        x = torch.cat((input_frame,broadcasted_action),1)

        x = F.relu(self.conv1(x))
        x = self.basic_block1(x)
        x = self.basic_block2(x)

        #output image head
        image_out = self.img_head(x)

        #output reward head
        reward_probability = self.reward_head(x)
        reward_out = torch.sum(reward_probability * self.reward_bins, 1)

        return image_out,reward_out




class MiniPacmanRGBToClassConverter():
    def __init__(self, use_cuda = True):
        self.use_cuda = use_cuda

        self.color_walls = torch.FloatTensor([1, 1, 1])  # 0
        self.color_food = torch.FloatTensor([0, 0, 1])  # 1
        self.color_pillman = torch.FloatTensor([0, 1, 0])  # 2
        self.color_ground = torch.FloatTensor([0, 0, 0]) # 3
        self.color_pill = torch.FloatTensor([0, 1, 1])  # 4

        self.color_ghost = torch.FloatTensor([1, 0, 0]) # 5
        self.color_ghost_edible = torch.FloatTensor([1, 1, 0]) # 6

        self.color = torch.stack([self.color_walls,
                                  self.color_food,
                                  self.color_pillman,
                                  self.color_ground,
                                  self.color_pill,
                                  self.color_ghost,
                                  self.color_ghost_edible])

        if use_cuda:
            self.color_walls = self.color_walls.cuda()
            self.color_food = self.color_food.cuda()
            self.color_pillman = self.color_pillman.cuda()
            self.color_ground = self.color_ground.cuda()
            self.color_pill = self.color_pill.cuda()
            self.color_ghost = self.color_ghost.cuda()
            self.color_ghost_edible = self.color_ghost_edible.cuda()
            self.color = self.color.cuda()

    def minipacman_rgb_to_class(self, state):
        state = state.view(state.shape[0], state.shape[2], state.shape[3], -1)

        # black python magic: if rgb colors are equal to walls color, food color etc.
        # -> we get an array of 3 times 1, so we can sum them in the 3th dimension up
        wall = ((state[:, :, :] == self.color_walls).sum(3) == 3).float()
        food = ((state[:, :, :] == self.color_food).sum(3) == 3).float()
        pillman = ((state[:, :, :] == self.color_pillman).sum(3) == 3).float()
        ground = ((state[:, :, :] == self.color_ground).sum(3) == 3).float()
        pill = ((state[:, :, :] == self.color_pill).sum(3) == 3).float()
        ghost = ((state[:, :, :] == self.color_ghost).sum(3) == 3).float()
        ghost_edible = ((state[:, :, :] == self.color_ghost_edible).sum(3) == 3).float()
        class_state = torch.stack([wall, food, pillman, ground, pill, ghost, ghost_edible], 1)

        if self.use_cuda:
            class_state = class_state.cuda()

        return class_state

    def minipacman_class_to_rgb(self, state):
        _, index = torch.max(state[:, :, :], 1)
        rgb_state_tensor =torch.index_select(self.color, 0, index.data.view(-1))
        rgb_state = rgb_state_tensor.view(state.shape[0],3,state.shape[2],state.shape[3])

        #legacy-code below serves as documentation for magic above
        '''for b in range(state.data.shape[0]):
            for x in range(state.data.shape[1]):
                for y in range(state.data.shape[2]):
                    _, class_state = torch.max(state[b, x, y], 0)
                    index = class_state.data[0]
                    if index == 0:
                        rgb_state[b, x, y] = self.color_walls
                    elif index == 1:
                        rgb_state[b, x, y] = self.color_food
                    elif index == 2:
                        rgb_state[b, x, y] = self.color_pillman
                    elif index == 3:
                        rgb_state[b, x, y] = self.color_ground
                    elif index == 5:
                        rgb_state[b, x, y] = self.color_pill
                    else:
                        rgb_state[b, x, y] = self.color_ghost'''
        return rgb_state

