import torch

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
        class_state = torch.stack([wall, food, pillman, ground, pill, ghost, ghost_edible], -1)

        if self.use_cuda:
            class_state = class_state.cuda()

        '''
        class_state = Variable(torch.zeros(state.data.shape[0], state.data.shape[1], state.data.shape[2], 6)).float()
        for b in range(state.data.shape[0]):
            for x in range(state.data.shape[1]):
                for y in range(state.data.shape[2]):
                    rgb = state[b, x, y]
                    if torch.equal(rgb.data, self.color_walls):
                        class_state[b, x, y, 0] = 1.
                    elif torch.equal(rgb.data, self.color_food):
                        class_state[b, x, y, 1] = 1.
                    elif torch.equal(rgb.data, self.color_pillman):
                        class_state[b, x, y, 2] = 1.
                    elif torch.equal(rgb.data, self.color_ground):
                        class_state[b, x, y, 3] = 1.
                    elif torch.equal(rgb.data, self.color_pill):
                        class_state[b, x, y, 5] = 1.
                    else:
                        class_state[b, x, y, 4] = 1.
        '''
        return class_state.view(class_state.shape[0], -1, class_state.shape[1], class_state.shape[2])

    def minipacman_class_to_rgb(self, state):
        state = state.view(state.shape[0], state.shape[2], state.shape[3], -1)

        _, index = torch.max(state[:, :, :], 3)
        rgb_state_tensor =torch.index_select(self.color, 0, index.view(-1))
        rgb_state = rgb_state_tensor.view(state.shape[0],3,state.shape[1],state.shape[2])

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

