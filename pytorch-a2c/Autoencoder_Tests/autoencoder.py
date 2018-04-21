import os
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Autoencoder_Tests.EncoderModel import EncoderModel
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
    render_window_sizes = (400, 400)
    cv2.namedWindow('target', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('target', render_window_sizes)
    cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('prediction', render_window_sizes)

    train_autoencoder(env_name="RegularMiniPacmanNoFrameskip-v0",
                      policy_model="RegularMiniPacmanNoFrameskip-v0.pt",
                      load_policy_model_dir="trained_models/a2c/",
                      root_path="/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c/",
                      save_encoder_model_freq=100,
                      save_encoder_model_dir="trained_autoencoder_models/",
                      save_encoder_model_name="RegularMiniPacmanAutoencoder_v1",
                      use_cuda=True)


def train_autoencoder(env_name="RegularMiniPacmanNoFrameskip-v0",
             policy_model = "POLICY_MODEL_NAME",
             load_policy_model_dir = "trained_models/",
             root_path="",
             save_encoder_model_freq = 100,
             save_encoder_model_dir = "",
             save_encoder_model_name = "",
             use_cuda=False):

    visdom_log_dir = "/tmp/gym/autoencoder"
    viz = Visdom()
    win = None
    try:
        os.makedirs(visdom_log_dir)
    except OSError:
        files = glob.glob(os.path.join(visdom_log_dir, '*.monitor.json'))
        for f in files:
            os.remove(f)

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    #create environment to train on
    #env = make_minipacman_env_no_log(env_name)
    make_method = make_minipacman_env(env_name,seed=1,rank=0,log_dir=visdom_log_dir)
    env = make_method()
    action_space = env.action_space.n

    #load policy to use to generate training states
    load_policy_model_dir = os.path.join(root_path, load_policy_model_dir)
    #policy = load_policy(load_policy_model_dir,
    #                     policy_model,
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
    encoder_model = EncoderModel(num_inputs = 4)
    if use_cuda:
        encoder_model.cuda()

    #initialize optimizer
    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.00001, weight_decay=1e-5)

    chance_of_random_action = 0.25

    #start training loop
    for i_episode in range(10000):
        #print("Start episode ",i_episode)
        state = env.reset()

        done = False
        game_step_counter = 0
        total_game_loss = 0

        while not done:
            game_step_counter += 1

            #encoder forward
            state_variable = Variable(torch.from_numpy(state).unsqueeze(0).type(FloatTensor))
            encoder_output = encoder_model(state_variable)
            criterion = torch.nn.MSELoss()
            loss = criterion(encoder_output, state_variable)
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
            render_observation(state_variable, encoder_output)

        print("Episode ",i_episode," loss: ", total_game_loss/game_step_counter)
        ##!!! currently logs something else --> TODO: log loss
        win = visdom_plot(viz, win,visdom_log_dir,"RegularMiniPacman","Autoencoder", smooth=0)

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

def render_observation_in_window(window_name, observation):
    drawable_state = observation[0][-1]
    drawable_state = drawable_state.data.cpu().numpy()

    frame_data = (drawable_state * 255.0)

    frame_data[frame_data < 0] = 0
    frame_data[frame_data > 255] = 255
    frame_data = frame_data.astype(np.uint8)

    cv2.imshow(window_name, frame_data)
    cv2.waitKey(1)


def render_observation(target, prediction):
    render_observation_in_window('target', target)
    render_observation_in_window('prediction', prediction)


def visdom_plot(viz, win, folder, game, name, bin_size=100, smooth=1):
    tx, ty = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))
    plt.xticks([1e2, 2e2, 4e2, 6e2, 8e2, 10e2],
               ["100", "200", "400", "600", "800", "1000"])
    plt.xlabel('Number of Episodes')
    plt.ylabel('Loss')

    plt.xlim(0, 10e2)

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    return viz.image(image, win=win)



if __name__ == '__main__':
    main()