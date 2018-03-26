import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
import cv2
import os
import logging
from tensorboardX import SummaryWriter
import time
from utils import torch_summarize

torch.manual_seed(42)


def get_step_summary(iteration, frame_loss, reward_loss, pred_reward, actual_reward, duration):

    # print table header again after every 3000th step
    if iteration % 3000 == 0 and iteration > 0:
        print('IT\t\tAVG Frame Loss\t AVG Reward Loss\tLST Prd. Rwrd\tLST Act Rwrd\tDuration')

    iteration_suffix = ''
    if iteration >= 1000:
        iteration /= 1000
        iteration_suffix = 'k'

    return '{0:.1f}{1}\t{2:.9f}\t\t{3:.5f}\t\t\t{4:.3f}\t\t\t{5:.3f}\t\t\t{6:.3f}'\
        .format(iteration, iteration_suffix, frame_loss, reward_loss, pred_reward, actual_reward, duration)


def init_weights(layer):
    """
    Initializes the weights of the given layer.
    Conv and Deconv layer will be initialized with a xavier initialization. Linear layers will
    be initialized uniform between -1 and 1
    :param layer:
    :return:
    """
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        layer.weight.data.uniform_(-1, 1)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform(layer.weight.data)


class PongEM(nn.Module):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}
    default_rms_prop = {"lr": 1e-4,
                        'alpha': 0.99,
                        'eps': 1e-08,
                        'weight_decay': 0,
                        'momentum': 0.9,
                        'centered': False}

    def __init__(self, name, env_state_dim=(1, 80, 80), num_input_actions=9, dropout=0.0, layer_init=True, enable_tensorboard=True):
        """
        :param env_state_dim:
        :param num_input_actions:
        :param dropout:
        :param enable_tensorboard:
        """

        super(PongEM, self).__init__()

        self.name = name

        # this logger will not print to the console. Only to the file.
        self.logger = logging.getLogger(__name__)

        # this logger will both print to the console as well as the file
        self.logger_prediction = logging.getLogger('prediction')

        env_st_channels, env_st_height, env_st_width = env_state_dim

        # since we also add the tiled actions we get 1 + 2 actions for the default pong case
        input_channels = env_st_channels

        # State Input
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=64,
                      kernel_size=8,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=6,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=0,
                      bias=True),
            nn.ReLU()
        )

        self.action_input = nn.Sequential(
            nn.Linear(in_features=num_input_actions,
                      out_features=1024
                      ),
            nn.ReLU(),
            nn.Linear(in_features=1024,
                      out_features=2048
                      ),
        )


        self.core_state_in = nn.Sequential(
            nn.Linear(in_features=1152,
                      out_features=2048
                      ),
            nn.ReLU(),
        )

        self.core_state_out = nn.Sequential(
            nn.Linear(in_features=2048,
                      out_features=1152
                      )
        )

        # Output Heads

        # Next State Head
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=6,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=6,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=256,
                               kernel_size=8,
                               stride=2,
                               padding=1,
                               bias=True)
        )

        # NOTE: needs 1 as output channels otherwise the state_size grows with each rollout iteration
        self.rewardHead = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 128, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_input_actions, bias=True),
            nn.Softmax(dim=1)
        )

        # Layer weight initalization
        if layer_init:
            for seq in self.children():
                for layer in seq.children():
                    init_weights(layer)

        weights = torch.FloatTensor([1.0])
        #self.loss_function_frame = nn.BCEWithLogitsLoss(weight=weights)

        #self.loss_function_frame = nn.NLLLoss()
        #self.loss_function_frame = nn.MSELoss()
        #self.loss_function_frame = nn.L1Loss()
        #self.loss_function_frame = nn.L2Loss()
        #self.loss_function_frame = nn.NLLLoss2d()
        self.loss_function_frame = nn.CrossEntropyLoss()
        self.loss_function_reward = nn.MSELoss()


        self.optimizer = torch.optim.Adam
        #self.optimizer = torch.optim.RMSprop

        self.use_cuda = False

        self.reward_loss_history = []
        self.frame_loss_history = []

        self.tb_writer = None

        model_summary = torch_summarize(self)

        if enable_tensorboard:
            self.tb_writer = SummaryWriter(comment=self.name)

            # for now until add graph works (needs pytorch version >= 0.4) add the model description as text
            self.tb_writer.add_text('model', model_summary, 0)

        self.logger.info('EM Model info:\n' + model_summary)
        self.logger.debug('Initialization finished.')
        print('Experiment name: ' + self.name)

    def forward(self, env_state_frame, env_input_action):  # input_action should be one_hot
        """

        :param env_state_frame: the current game state as a 1, 1, 80, 80 frame scaled to [0, 1]
        :param env_input_action: the current action that the agent takes
        :return: (env_state_frame_output, env_state_reward_output) -> new output state & new reward
        """

        # Convolution Layers
        state_conv = self.conv(env_state_frame)
        # Storing the output size of the conv layers to ensure the output layer will have the same size
        shape_out = state_conv.size()
        state_conv = state_conv.view(state_conv.size(0), -1)
        state_conv = self.core_state_in(state_conv)

        # Action input layers
        fc_action = self.action_input(env_input_action)
        if fc_action.size(0) == 2048:
            fc_action = fc_action.view((1, 2048))
            state_conv = state_conv.view((1, 2048))

        # Core Layers
        core = state_conv
        core = state_conv * fc_action
        state_deconv = self.core_state_out(core)

        # Deconv Layers
        #   These layers are responsible for generating the next state
        state_deconv = state_deconv.view(shape_out)
        state_deconv = self.deconv(state_deconv)

        # scale the output (-> predicted frame) to [0, 1]
        out_min, out_max = state_deconv.min(), state_deconv.max()
        env_state_frame_output = state_deconv
        #env_state_frame_output = (state_deconv - out_min) / (out_max - out_min)

        # Reward Layers
        #   These layers are responsible for calculating the reward of the
        env_state_reward_output = self.rewardHead(core)

        return env_state_frame_output, env_state_reward_output

    def _step(self, env_state_frame, env_action, env_state_frame_target, env_reward_target):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        self.optimizer.zero_grad()
        _, executed_action_index = env_action.max(0)

        # Compute loss and gradient
        next_frame, next_reward = self.forward(env_state_frame, env_action)

        #env_state_frame_target.data = 1 - env_state_frame_target.data
        #out_min, out_max = env_state_frame_target.min(), env_state_frame_target.max()
        #env_state_frame_target = (env_state_frame_target - out_min) / (out_max - out_min)
        next_frame_loss = self.loss_function_frame(next_frame, env_state_frame_target[0])

        next_reward_loss = self.loss_function_reward(next_reward[0][executed_action_index], env_reward_target)

        # preform training step with both losses
        losses = [next_frame_loss, next_reward_loss]
        grad_seq = [losses[0].data.new(1).fill_(1) for _ in range(len(losses))]

        torch.autograd.backward(losses, grad_seq)

        self.optimizer.step()

        # for debugging and diagnose purposes return loss frame and reward
        # TODO Check if this section is needed later on
        # TODO Add post processing of the frame
        # next_frame = 1 - next_frame
        return (next_frame_loss, next_reward_loss), (next_frame, next_reward)

    # TODO: base class method signature is (self, mode=True). Consider to change this signature to overwrite base class
    def train(self,
              agent,
              num_iterations=100,
              mode=True,
              optimizer_args={},
              should_use_cuda=False,
              render=True,
              render_window_sizes=(400, 400),
              save_images_after_xth_iteration=-1,
              save_images_every_xth_iteration=37,
              log_every_xth_iteration=100):

        # initialize optimizer
        self.optimizer = self.optimizer(self.parameters(), **optimizer_args)
        save_images_path = os.path.join(os.getcwd(), 'logs', self.name, 'images')
        # possibility to break out of loop
        continue_training = True

        if torch.cuda.is_available() and should_use_cuda:
            self.cuda()
            self.logger.debug('train with cuda support')
            self.use_cuda = True

        # if we render the images make the window bigger so that it is easier to see what is happening
        if render:
            self.logger.debug('Render is true')
            cv2.namedWindow('frame_prediction')#, cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame_prediction', render_window_sizes[0], render_window_sizes[1])
            cv2.namedWindow('frame_target')#, cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame_target', render_window_sizes[0], render_window_sizes[1])

        self.logger.debug('Optimizer Args: ' + str(optimizer_args))
        self.logger.info('START Training with {0} iterations.'.format(num_iterations))

        self.logger_prediction.info('IT\t\tAVG Frame Loss\t AVG Reward Loss\tLST Prd. Rwrd\tLST Act Rwrd\tDuration')
        self.logger_prediction.info('=====================================================================================================')
        start_time = time.time()

        for iteration in range(num_iterations):
            # should we stop training?
            if not continue_training:
                self.logger.info('STOP training.')
                break

            # get training data from agent.
            # for the EM we need
            #   - the current state (game frame)
            #   - the current action
            #   - the frame for the next step
            #   - the reward for the next step
            env_state_frame, env_action, env_reward_target, env_state_frame_target = agent.play_step()
            action_index = np.argmax(env_action)

            if render:
                frame_data = env_state_frame[0][0]
                cv2.imshow('frame_target', frame_data)
                if save_images_after_xth_iteration != -1 \
                        and iteration > save_images_after_xth_iteration \
                        and iteration % save_images_every_xth_iteration == 0:
                    cv2.imwrite(os.path.join(save_images_path, '{0}_f.png'.format(iteration)), frame_data)

            if self.use_cuda:
                env_state_frame = Variable(torch.from_numpy(env_state_frame)).float().cuda()
                env_action = Variable(torch.from_numpy(env_action)).float().cuda()
                env_state_frame_target = Variable(torch.from_numpy(env_state_frame_target)).long().cuda()
                env_reward_target = Variable(torch.from_numpy(env_reward_target)).float().cuda()
            else:
                env_state_frame = Variable(torch.from_numpy(env_state_frame)).float()
                env_action = Variable(torch.from_numpy(env_action)).float()
                env_state_frame_target = Variable(torch.from_numpy(env_state_frame_target)).long()
                env_reward_target = Variable(torch.from_numpy(env_reward_target)).float()

            # run the models forward pass
            (next_frame_loss, next_reward_loss), (next_frame_prediction, next_reward_prediction) = self._step(
                env_state_frame,
                env_action,
                env_state_frame_target,
                env_reward_target)

            if render:
                _, next_frame_prediction = torch.max(next_frame_prediction, 1)
                frame_data = next_frame_prediction.cpu().data.numpy()[0].astype(np.uint8)

                cv2.imshow('frame_prediction', frame_data)
                cv2.waitKey(1)

                if save_images_after_xth_iteration != -1 \
                        and iteration > save_images_after_xth_iteration \
                        and iteration % save_images_every_xth_iteration == 0:
                    # prepare for saving
                    frame_data *= 255
                    frame_data = frame_data.astype(np.uint8)
                    cv2.imwrite(os.path.join(save_images_path, '{0}_f_h.png'.format(iteration)), frame_data)

            if self.tb_writer is not None:
                self.tb_writer.add_scalar('data/em/loss_frame', next_frame_loss, iteration)
                self.tb_writer.add_scalar('data/em/loss_reward', next_reward_loss, iteration)

                # log current frame prediction every 100 iterations
                if iteration % 100 == 0:
                    images = vutils.make_grid(next_frame_prediction.data[0], normalize=True, scale_each=True)
                    self.tb_writer.add_image('frame_predictions', images, iteration)

            self.frame_loss_history.append(next_frame_loss.data.cpu().numpy()[0])
            self.reward_loss_history.append(next_reward_loss.data.cpu().numpy()[0])

            if iteration % log_every_xth_iteration == 0 and iteration > 0:
                duration = time.time() - start_time
                start_time = time.time()

                next_reward_prediction = next_reward_prediction.cpu().data[0][action_index]
                env_reward = env_reward_target.cpu().data[0]

                # get mean losses
                last_frame_losses = np.array(self.frame_loss_history[iteration - log_every_xth_iteration:iteration])
                last_reward_losses = np.array(self.reward_loss_history[iteration - log_every_xth_iteration:iteration])
                self.logger_prediction.info(get_step_summary(iteration, last_frame_losses.mean(), last_reward_losses.mean(), next_reward_prediction, env_reward, duration))

        if self.tb_writer is not None:
            self.tb_writer.export_scalars_to_json(os.path.join(os.getcwd(), 'logs', self.name, "em_all_scalars.json"))
            self.tb_writer.close()

    def _tile_and_concat_input(self, env_state_frame, env_input_action):
        # TODO: This is not pretty!
        if self.use_cuda:
            env_input_action = \
                Variable(torch.zeros(
                    env_state_frame.data.shape[2],
                    env_state_frame.data.shape[3],
                    env_state_frame.data.shape[0])
                ).float().cuda() + env_input_action
        else:
            env_input_action = \
                Variable(torch.zeros(
                    env_state_frame.data.shape[2],
                    env_state_frame.data.shape[3],
                    env_state_frame.data.shape[0])
                ).float() + env_input_action

        env_input_action = torch.unsqueeze(env_input_action, 0).permute(0, 3, 1, 2)
        x = torch.cat([env_state_frame, env_input_action], 1)
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
