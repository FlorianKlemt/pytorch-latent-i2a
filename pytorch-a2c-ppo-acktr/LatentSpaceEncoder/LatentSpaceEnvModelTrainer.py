import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
from LatentSpaceEncoder.LossPrinter import LossPrinter


class LatentSpaceEnvModelTrainer():
    def __init__(self, auto_encoder_model, env_encoder_model, loss_criterion, auto_optimizer, next_pred_optimizer, use_cuda, visualize,
                 use_only_last_frame_of_state, grey_scale):
        self.auto_encoder_model = auto_encoder_model
        self.env_encoder_model = env_encoder_model
        self.loss_criterion = loss_criterion
        self.auto_optimizer = auto_optimizer
        self.next_pred_optimizer = next_pred_optimizer
        self.use_cuda = use_cuda
        self.visualize = visualize
        self.use_only_last_frame_of_state = use_only_last_frame_of_state
        self.grey_scale = grey_scale
        if self.visualize:
            render_window_sizes = (400, 400)
            cv2.namedWindow('predicted', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('predicted', render_window_sizes)
            cv2.namedWindow('next_ground_truth', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('next_ground_truth', render_window_sizes)
            cv2.namedWindow('autoencoder', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('autoencoder', render_window_sizes)
            cv2.namedWindow('substracted_auto_encoder', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('substracted_auto_encoder', render_window_sizes)
            cv2.namedWindow('current_ground_truth', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('current_ground_truth', render_window_sizes)


    def train_env_model_step(self, first_state_variable, second_state_variable, action):
        target = first_state_variable
        first_state_prediction = self.auto_encoder_model(target)

        first_state_loss = self.loss_criterion(first_state_prediction, target)

        # first state encoder backward
        self.auto_optimizer.zero_grad()
        first_state_loss.backward()
        self.auto_optimizer.step()


        # first state encode in latent space
        first_state_latent_prediction = self.auto_encoder_model.encode(target)

        # second state encode in latent space
        '''if self.use_only_last_frame_of_state:
            second_state_latent_prediction = self.auto_encoder_model.encode(second_state_variable[0][-1])
        else:
            second_state_latent_prediction = self.auto_encoder_model.encode(second_state_variable[0])'''
        second_state_latent_prediction = self.auto_encoder_model.encode(second_state_variable)



        # first-to-second forward
        action_variable = Variable(torch.from_numpy(np.array([action])).float())
        if self.use_cuda:
            action_variable = action_variable.cuda()
        latent_prediction = self.env_encoder_model(first_state_latent_prediction, action_variable)
        latent_target = second_state_latent_prediction
        latent_target = Variable(latent_target.data, requires_grad=False)
        latent_loss = self.loss_criterion(latent_prediction, latent_target)

        #first-to-second backward
        self.next_pred_optimizer.zero_grad()
        latent_loss.backward()
        self.next_pred_optimizer.step()

        if self.visualize:
            # render last of the frame_stack for ground truth and for encoder
            decoded_prediction = self.auto_encoder_model.decode(latent_prediction)
            #render_observation_in_window('predicted', decoded_prediction[-1], None) #print last frame of decoded prediction
            #render_observation_in_window('next_ground_truth', second_state_variable[0][-1], None)
            #render_observation_in_window('autoencoder', first_state_prediction[-1], None)   #print last frame of first state prediction
            #render_observation_in_window('current_ground_truth', first_state_variable[0][-1], None)
            #render_observation_in_window('substracted_auto_encoder',first_state_variable[0][-1]-first_state_prediction[-1] , None)

            #for rgb without framestack
            render_observation_in_window('predicted', decoded_prediction, None, grey_scale=self.grey_scale)
            render_observation_in_window('next_ground_truth', second_state_variable, None, grey_scale=self.grey_scale)
            render_observation_in_window('autoencoder', first_state_prediction,None, grey_scale=self.grey_scale)
            render_observation_in_window('current_ground_truth', first_state_variable, None, grey_scale=self.grey_scale)
            render_observation_in_window('substracted_auto_encoder', first_state_variable - first_state_prediction, None, grey_scale=self.grey_scale)

        return first_state_loss, latent_loss


    def train_env_encoder(self, env, policy, use_cuda):
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        loss_printer = LossPrinter()
        chance_of_random_action = 0.25
        for i_episode in range(10000):
            loss_printer.reset()
            state = env.reset()

            done = False
            while not done:
                first_state_variable = Variable(torch.from_numpy(state).unsqueeze(0).type(FloatTensor))

                # let policy decide on next action
                critic, actor = policy(first_state_variable)

                action = sample_action_from_distribution(actor=actor, action_space=env.action_space.n,
                                                         chance_of_random_action=chance_of_random_action)

                # perform action to get next state
                next_state, reward, done, _ = env.step(action)

                second_state_variable = Variable(torch.from_numpy(next_state).unsqueeze(0).type(FloatTensor))
                first_state_loss, latent_loss = self.train_env_model_step(
                    first_state_variable=first_state_variable, second_state_variable=second_state_variable,
                    action=action)

                loss_printer.add_loss(first_state_loss.data[0], latent_loss.data[0])

                state = next_state

            loss_printer.print_episode(i_episode=i_episode)


    def train_env_encoder_batchwise(self, env, policy, use_cuda):
        from collections import deque
        sample_memory = deque(maxlen=2000)

        loss_printer = LossPrinter()
        chance_of_random_action = 0.25
        for i_episode in range(10000):
            loss_printer.reset()
            state = env.reset()
            state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
            if use_cuda:
                state = state.cuda()

            done = False
            while not done:
                # let policy decide on next action and perform it
                critic, actor = policy(state)
                action = sample_action_from_distribution(actor=actor, action_space=env.action_space.n,
                                                         chance_of_random_action=chance_of_random_action)
                next_state, reward, done, _ = env.step(action)
                next_state = Variable(torch.from_numpy(next_state).unsqueeze(0)).float()
                if use_cuda:
                    next_state = next_state.cuda()

                # add current state, next-state pair to replay memory
                sample_memory.append((state, next_state, action))

                # sample a state, next-state pair randomly from replay memory for a training step
                sample_state, sample_next_state, sample_action = random.choice(sample_memory)
                first_state_loss, latent_loss = self.train_env_model_step(
                    first_state_variable=sample_state,
                    second_state_variable=sample_next_state,
                    action=sample_action)

                loss_printer.add_loss(first_state_loss.data[0], latent_loss.data[0])

                state = next_state

            loss_printer.print_episode(i_episode=i_episode)


def sample_action_from_distribution(actor, action_space, chance_of_random_action=0.25):
    prob = F.softmax(actor, dim=1)
    action = prob.multinomial().data
    action = action[0][0]
    if random.random() < chance_of_random_action:
        action = random.randint(0, action_space - 1)
    return action


import numpy as np
import cv2
def render_observation_in_window(window_name, observation, mean_image=None, grey_scale=False):
    if grey_scale:
        drawable_state = observation.view(-1, 1, observation.data.shape[2], observation.data.shape[3])[-1]
    else:
        drawable_state = observation.view(-1, 3, observation.data.shape[2], observation.data.shape[3])[-1]
    drawable_state = drawable_state.data.cpu().numpy()

    if mean_image is not None:
        drawable_state -= mean_image

    frame_data = (drawable_state * 255.0)

    frame_data[frame_data < 0] = 0
    frame_data[frame_data > 255] = 255
    frame_data = frame_data.astype(np.uint8)

    #why the heck would cv2 store images in BGR per default??
    #convert to rgb + transpose because cv2 wants images in the form (x,y,channels)
    if grey_scale:
        cv2_frame_data = np.transpose(frame_data, (1, 2, 0))
    else:
        cv2_frame_data = cv2.cvtColor(np.transpose(frame_data, (1, 2, 0)), cv2.COLOR_BGR2RGB)

    cv2.imshow(window_name, cv2_frame_data)
    cv2.waitKey(1)