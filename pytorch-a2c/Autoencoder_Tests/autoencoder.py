import os
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Autoencoder_Tests.AutoEncoderModel import AutoEncoderModel
from A2C_Models.MiniModel import MiniModel
from minipacman_envs import make_minipacman_env_no_log, make_minipacman_env
import gym_minipacman
import cv2
import numpy as np
import glob

from visualize_atari import load_data
from visdom import Visdom
import matplotlib.pyplot as plt

def main():
    #magic do not change
    plt.switch_backend('TKAgg')
    plt.ion()
    #matplotlib init
    smooth_loss_plot, loss_plot = init_loss_plot()

    #cv init
    render_window_sizes = (400, 400)
    cv2.namedWindow('target', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('target', render_window_sizes)
    cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('prediction', render_window_sizes)
    cv2.namedWindow('target_substract_mean', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('target_substract_mean', render_window_sizes)
    cv2.namedWindow('prediction_substract_mean', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('prediction_substract_mean', render_window_sizes)

    use_cuda = True

    env, encoder_model, policy = init_autoencoder_training(env_name="RegularMiniPacmanNoFrameskip-v0",
                              root_path="/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c/",
                              policy_model_name="RegularMiniPacmanNoFrameskip-v0.pt",
                              load_policy_model_dir="trained_models/a2c/",
                              use_cuda=use_cuda)

    mean_image = get_mean_image(env=env,policy=policy,use_cuda=use_cuda)
    mean_image = np.squeeze(mean_image)

    loss_criterion = torch.nn.MSELoss()
    #loss_criterion = torch.nn.BCEWithLogitsLoss()  # BCELoss gives weird cuda device assert error

    train_autoencoder(env=env,
                      encoder_model=encoder_model,
                      policy=policy,
                      loss_criterion=loss_criterion,
                      mean_image=mean_image,
                      save_encoder_model_freq=100,
                      save_encoder_model_dir="trained_autoencoder_models/",
                      save_encoder_model_name="RegularMiniPacmanAutoencoder_v1",
                      use_cuda=use_cuda,
                      loss_plot=loss_plot,
                      smooth_loss_plot=smooth_loss_plot)


def get_mean_image(env,policy,use_cuda):
    frame_list = []
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    state = env.reset()
    games_to_play = 500
    p = 0.2
    for i in range(games_to_play):
        print("Game ",i)
        done = False
        while not done:
            if random.random() < p:
                frame_list.append(state[-1]) #append last frame of the frame stack of the current state to frame_list
            state_variable = Variable(torch.from_numpy(state).unsqueeze(0).type(FloatTensor))
            critic, actor = policy(state_variable)
            prob = F.softmax(actor, dim=1)
            action = prob.multinomial().data
            action = action[0][0]
            next_state, reward, done, _ = env.step(action)
            state = next_state

    N = len(frame_list)
    print(N)

    arr = np.zeros((19, 19, 1), np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in frame_list:
        imarr = np.array(im, dtype=np.float)
        imarr = np.expand_dims(imarr, axis=2)
        arr = arr + imarr / N

    # draw resulting mean image
    drawable_state = arr
    frame_data = (drawable_state * 255.0)
    frame_data = frame_data.astype(np.uint8)
    cv2.imshow('target', frame_data)
    cv2.waitKey(0)

    return arr


def init_autoencoder_training(env_name, root_path, load_policy_model_dir, policy_model_name, use_cuda):
    #create environment to train on
    env = make_minipacman_env_no_log(env_name)
    #make_method = make_minipacman_env(env_name,seed=1,rank=0,log_dir=visdom_log_dir)
    #env = make_method()
    action_space = env.action_space.n

    #load policy to use to generate training states
    load_policy_model_dir = os.path.join(root_path, load_policy_model_dir)
    #policy = load_policy(load_policy_model_dir,
    #                     policy_model_name,
    #                     action_space=action_space,
    #                     use_cuda=use_cuda)
    policy = MiniModel(num_inputs=4, action_space=action_space, use_cuda=use_cuda)
    if use_cuda:
        policy.cuda()

    #generate new autoencoder to train, or load an existing one to resume training
    #save_environment_model_dir = os.path.join(root_path, save_environment_model_dir)
    #if load_autoencoder_model:
    #    load_environment_model_dir = os.path.join(root_path, load_environment_model_dir)
    #    encoder_model = load_encoder(EMModel,
    #                                      load_environment_model_dir,
    #                                      environment_model_name,
    #                                      action_space,
    #                                      use_cuda)
    #else:
    encoder_model = AutoEncoderModel(num_inputs = 1)
    if use_cuda:
        encoder_model.cuda()
    return env, encoder_model, policy


def train_autoencoder(
             env = None,
             encoder_model = None,
             policy = None,
             loss_criterion = torch.nn.MSELoss(),
             mean_image=None,
             save_encoder_model_freq = 100,
             save_encoder_model_dir = "",
             save_encoder_model_name = "",
             use_cuda=False,
             loss_plot=None,
             smooth_loss_plot=None):

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    mean_image_variable = Variable(torch.from_numpy(mean_image).type(FloatTensor))

    #initialize optimizer
    #optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.00001, weight_decay=1e-5)
    optimizer = torch.optim.RMSprop(encoder_model.parameters(), lr=0.00001, weight_decay=1e-5)

    chance_of_random_action = 0.25

    #start training loop
    loss_list = []
    for i_episode in range(10000):
        state = env.reset()

        done = False
        game_step_counter = 0
        total_game_loss = 0

        while not done:
            game_step_counter += 1

            #encoder forward
            state_variable = Variable(torch.from_numpy(state).unsqueeze(0).type(FloatTensor))
            #TODO: only last frame
            encoder_output = encoder_model(state_variable[0][-1])

            if mean_image is not None:
                prediction = encoder_output - mean_image_variable
                target = state_variable[0][-1] - mean_image_variable
            else:
                # make loss count only on the last frame of the frame stack
                prediction = encoder_output
                target = state_variable[0][-1]
                # prediction=encoder_output, target=state_variable)       #make loss count on the full frame stack

            loss = loss_criterion(prediction, target)
            total_game_loss += loss.data[0]

            #encoder backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            #let policy decide on next action
            critic, actor = policy(state_variable)

            prob = F.softmax(actor, dim=1)
            action = prob.multinomial().data
            action = action[0][0]
            if random.random() < chance_of_random_action:
                action = random.randint(0, env.action_space.n -1)

            #perform action to get next state
            next_state, reward, done, _ = env.step(action)
            state = next_state

            #render last of the frame_stack for ground truth and for encoder
            render_observation(state_variable[0][-1], encoder_output, mean_image)

        print("Episode ",i_episode," loss: ", total_game_loss/game_step_counter)
        loss_list.append(loss.data[0])
        plot_loss(i_episode,loss_list,smooth_loss_plot)
        plot_smooth_loss(i_episode,loss_list,loss_plot)

        if i_episode % save_encoder_model_freq == 0:
            print("Save model", save_encoder_model_dir, save_encoder_model_name)
            save_encoder_model(save_model_dir = save_encoder_model_dir,
                                   encoder_model_name = save_encoder_model_name,
                                   encoder_model = encoder_model)

def save_encoder_model(save_model_dir, encoder_model_name, encoder_model):
    state_to_save = encoder_model.state_dict()
    save_model_path = '{0}{1}.dat'.format(save_model_dir, encoder_model_name)
    torch.save(state_to_save, save_model_path)


def load_policy(load_policy_model_dir="trained_models/",
                policy_file=None,
                action_space=None,
                use_cuda=True):
    saved_state = torch.load('{0}{1}'.format(
        load_policy_model_dir, policy_file), map_location=lambda storage, loc: storage)

    policy_model = MiniModel(num_inputs=4, action_space=action_space, use_cuda=use_cuda)
    policy_model.load_state_dict(saved_state)
    if use_cuda:
        policy_model.cuda()

    for param in policy_model.parameters():
        param.requires_grad = False

    policy_model.eval()
    return policy_model



def render_observation_in_window(window_name, observation, mean_image=None):
    drawable_state = observation#[0][-1]
    drawable_state = drawable_state.data.cpu().numpy()

    if mean_image is not None:
        drawable_state -= mean_image

    frame_data = (drawable_state * 255.0)

    frame_data[frame_data < 0] = 0
    frame_data[frame_data > 255] = 255
    frame_data = frame_data.astype(np.uint8)

    cv2.imshow(window_name, frame_data)
    cv2.waitKey(1)


def render_observation(target, prediction,mean_image):
    if mean_image is not None:
        render_observation_in_window('target_substract_mean', target, mean_image)
        render_observation_in_window('prediction_substract_mean', prediction, mean_image)
    render_observation_in_window('target', target, None)
    render_observation_in_window('prediction', prediction, None)



def init_loss_plot():
    fig = plt.figure()

    smooth_loss_plot = plt.subplot(121)
    smooth_loss_plot.set_yscale('log')     #gca for get_current_axis
    plt.xlabel('Number of Episodes')
    plt.ylabel('Smoothed Loss')
    plt.autoscale(enable=True, axis='x', tight=None)
    plt.title("MiniPacman")
    plt.legend(loc=4)

    loss_plot = plt.subplot(122)
    loss_plot.set_yscale('log')  # gca for get_current_axis
    plt.xlabel('Number of Episodes')
    plt.ylabel('Loss')
    plt.autoscale(enable=True, axis='x', tight=None)
    plt.title("MiniPacman")
    plt.legend(loc=4)
    return smooth_loss_plot, loss_plot

def plot_smooth_loss(episode, loss_list, subplot):
    subplot.cla()
    #plt.cla()
    k = list(range(0, episode+1))
    plot_list = [0]*len(loss_list)
    for i in range(len(loss_list)):
        plot_list[i] = np.mean(loss_list[max(i-5,0):min(i+5,len(loss_list))])
    subplot.plot(k, plot_list, 'r')
    plt.show()
    plt.pause(0.001)

def plot_loss(episode, loss_list, subplot):
    k = list(range(0, episode+1))
    subplot.plot(k, loss_list, 'r')
    plt.show()
    plt.pause(0.001)



if __name__ == '__main__':
    main()