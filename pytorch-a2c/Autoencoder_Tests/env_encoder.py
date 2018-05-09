import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
from Autoencoder_Tests.EnvEncoderModel import EnvEncoderModel
import matplotlib.pyplot as plt
import cv2
import gym


def main():
    render_window_sizes = (400, 400)
    cv2.namedWindow('predicted', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('predicted', render_window_sizes)
    cv2.namedWindow('next_ground_truth', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('next_ground_truth', render_window_sizes)
    cv2.namedWindow('autoencoder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('autoencoder', render_window_sizes)

    #magic do not change
    plt.switch_backend('TKAgg')
    plt.ion()

    use_cuda = True
    from A2C_Models.model import ActorCritic
    from A2C_Models.MiniModel import MiniModel

    latent_space = 32#64
    encoder_space = 128#128
    hidden_space = 128#128

    env, auto_encoder_model, policy = init_autoencoder_training("MsPacman-v0",#env_name="PongDeterministic-v4",
                                                                policy_model_alias=ActorCritic,
                                                                policy_model_input_channels=4,
                                                                input_size=(210,160),#input_size=(84,84),
                                                                latent_space=latent_space,
                                                                hidden_space = hidden_space,
                                                                use_cuda=use_cuda)

    #env, auto_encoder_model, policy = init_autoencoder_training(env_name="RegularMiniPacmanNoFrameskip-v0",
    #                                                            #root_path="/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c/",
    #                                                            #policy_model_name="RegularMiniPacmanNoFrameskip-v0.pt",
    #                                                            #load_policy_model_dir="trained_models/a2c/",
    #                                                            policy_model_alias=MiniModel,
    #                                                            policy_model_input_channels=4,
    #                                                            input_size=(19,19),
    #                                                            use_cuda=use_cuda)

    env_encoder_model = EnvEncoderModel(num_inputs=1, latent_space=latent_space, encoder_space=encoder_space, action_broadcast_size=50, use_cuda=use_cuda)
    if use_cuda:
        env_encoder_model.cuda()
    loss_criterion = torch.nn.MSELoss()
    auto_optimizer = torch.optim.RMSprop(auto_encoder_model.parameters(), lr=0.00005, weight_decay=1e-5)             #0.0001!!!
    env_encoder_optimizer = torch.optim.RMSprop(env_encoder_model.parameters(), lr=0.00005, weight_decay=1e-5)       #0.0001!!!

    latent_space_trainer = LatentSpaceEnvModelTrainer(auto_encoder_model, env_encoder_model, loss_criterion, auto_optimizer, env_encoder_optimizer, use_cuda)
    train_env_encoder(env, policy, latent_space_trainer, use_cuda)
    #train_env_encoder_batchwise(env, policy, latent_space_trainer, use_cuda)



def init_autoencoder_training(env_name, policy_model_alias, policy_model_input_channels, input_size, latent_space, hidden_space, use_cuda):
    from Autoencoder_Tests.AutoEncoderModel import AutoEncoderModel
    from minipacman_envs import make_minipacman_env_no_log
    #create environment to train on
    if "MiniPacman" in env_name:
        env = make_minipacman_env_no_log(env_name)
    else:
        from envs import WrapPyTorch
        from baselines.common.atari_wrappers import wrap_deepmind, FrameStack
        env = WarpFrameGrayScale(gym.make(env_name))
        env = WrapPyTorch(FrameStack(env, 4))
        #env = WrapPyTorch(wrap_deepmind(gym.make(env_name)))

    action_space = env.action_space.n

    policy = policy_model_alias(num_inputs=policy_model_input_channels, action_space=action_space, use_cuda=use_cuda)
    if use_cuda:
        policy.cuda()

    encoder_model = AutoEncoderModel(num_inputs = 1, input_size=input_size, latent_space=latent_space, hidden_space=hidden_space)
    if use_cuda:
        encoder_model.cuda()
    return env, encoder_model, policy


def sample_action_from_distribution(actor, action_space, chance_of_random_action=0.25):
    prob = F.softmax(actor, dim=1)
    action = prob.multinomial().data
    action = action[0][0]
    if random.random() < chance_of_random_action:
        action = random.randint(0, action_space - 1)
    return action

def train_env_encoder(env, policy, latent_space_trainer, use_cuda):
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

            action = sample_action_from_distribution(actor=actor, action_space=env.action_space.n, chance_of_random_action=chance_of_random_action)

            # perform action to get next state
            next_state, reward, done, _ = env.step(action)

            second_state_variable = Variable(torch.from_numpy(next_state).unsqueeze(0).type(FloatTensor))
            first_state_loss, latent_loss = latent_space_trainer.train_env_model_step(first_state_variable=first_state_variable, second_state_variable=second_state_variable, action=action)

            loss_printer.add_loss(first_state_loss.data[0], latent_loss.data[0])

            state = next_state

        loss_printer.print_episode(i_episode=i_episode)


def train_env_encoder_batchwise(env, policy, latent_space_trainer, use_cuda):
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
            action = sample_action_from_distribution(actor=actor, action_space=env.action_space.n, chance_of_random_action=chance_of_random_action)
            next_state, reward, done, _ = env.step(action)
            next_state = Variable(torch.from_numpy(next_state).unsqueeze(0)).float()
            if use_cuda:
                next_state = next_state.cuda()

            # add current state, next-state pair to replay memory
            sample_memory.append((state,next_state,action))

            #sample a state, next-state pair randomly from replay memory for a training step
            sample_state, sample_next_state, sample_action = random.choice(sample_memory)
            first_state_loss, latent_loss = latent_space_trainer.train_env_model_step(first_state_variable=sample_state,
                                                                                      second_state_variable=sample_next_state,
                                                                                      action=sample_action)

            loss_printer.add_loss(first_state_loss.data[0], latent_loss.data[0])

            state = next_state

        loss_printer.print_episode(i_episode=i_episode)



import numpy as np
class LossPrinter():
    def __init__(self):
        self.auto_loss_list = []
        self.latent_pred_loss_list = []
        self.reset()
        self.auto_loss_plot, self.latent_pred_loss_plot = self.init_loss_plot()

    def reset(self):
        self.game_step_counter = 0
        self.total_game_auto_loss = 0
        self.total_game_pred_loss = 0

    def add_loss(self, first_state_loss, latent_loss):
        self.auto_loss_list.append(first_state_loss)
        self.latent_pred_loss_list.append(latent_loss)
        self.total_game_auto_loss += first_state_loss
        self.total_game_pred_loss += latent_loss
        self.game_step_counter += 1

    def print_episode(self, i_episode):
        print("Episode ", i_episode, " auto_loss: ", self.total_game_auto_loss / self.game_step_counter, " pred_loss: ",
              self.total_game_pred_loss / self.game_step_counter)

    def plot_loss(self, i_episode):
        self.plot_smooth_loss(i_episode, self.auto_loss_list, self.auto_loss_plot)
        self.plot_smooth_loss(i_episode, self.latent_pred_loss_list, self.latent_pred_loss_plot)

    def plot_smooth_loss(self, episode, loss_list, subplot):
        subplot.cla()
        k = list(range(0, episode + 1))
        plot_list = [0] * len(loss_list)
        for i in range(len(loss_list)):
            plot_list[i] = np.mean(loss_list[max(i - 5, 0):min(i + 5, len(loss_list))])
        subplot.plot(k, plot_list, 'r')
        plt.show()
        plt.pause(0.001)

    def plot_non_smooth_loss(self, episode, loss_list, subplot):
        k = list(range(0, episode + 1))
        subplot.plot(k, loss_list, 'r')
        plt.show()
        plt.pause(0.001)

    def init_loss_plot(self):
        fig = plt.figure()

        auto_loss_plot = plt.subplot(121)
        auto_loss_plot.set_yscale('log')  # gca for get_current_axis
        plt.xlabel('Number of Episodes')
        plt.ylabel('Smoothed Loss')
        plt.autoscale(enable=True, axis='x', tight=None)
        plt.title("Autoencoder Loss")
        plt.legend(loc=4)

        latent_pred_loss_plot = plt.subplot(122)
        latent_pred_loss_plot.set_yscale('log')  # gca for get_current_axis
        plt.xlabel('Number of Episodes')
        plt.ylabel('Latent Space Prediction Loss')
        plt.autoscale(enable=True, axis='x', tight=None)
        plt.title("MiniPacman")
        plt.legend(loc=4)
        return auto_loss_plot, latent_pred_loss_plot






class LatentSpaceEnvModelTrainer():
    def __init__(self, auto_encoder_model, env_encoder_model, loss_criterion, auto_optimizer, next_pred_optimizer, use_cuda):
        self.auto_encoder_model = auto_encoder_model
        self.env_encoder_model = env_encoder_model
        self.loss_criterion = loss_criterion
        self.auto_optimizer = auto_optimizer
        self.next_pred_optimizer = next_pred_optimizer
        self.use_cuda = use_cuda


    def train_env_model_step(self, first_state_variable, second_state_variable, action):
        # first state encoder forward
        target = first_state_variable[0][-1]
        first_state_prediction = self.auto_encoder_model(target)

        first_state_loss = self.loss_criterion(first_state_prediction, target)

        # first state encoder backward
        self.auto_optimizer.zero_grad()
        first_state_loss.backward()
        self.auto_optimizer.step()


        # first state encode in latent space
        first_state_latent_prediction = self.auto_encoder_model.encode(target)

        # second state encode in latent space
        second_state_latent_prediction = self.auto_encoder_model.encode(second_state_variable[0][-1])



        # first-to-second forward
        latent_prediction = self.env_encoder_model(first_state_latent_prediction, action)
        latent_target = second_state_latent_prediction
        latent_target = Variable(latent_target.data, requires_grad=False)
        latent_loss = self.loss_criterion(latent_prediction, latent_target)

        #first-to-second backward
        self.next_pred_optimizer.zero_grad()
        latent_loss.backward()
        self.next_pred_optimizer.step()

        # render last of the frame_stack for ground truth and for encoder
        decoded_prediction = self.auto_encoder_model.decode(latent_prediction)
        render_observation_in_window('predicted', decoded_prediction, None)
        render_observation_in_window('next_ground_truth', second_state_variable[0][-1], None)
        render_observation_in_window('autoencoder', first_state_prediction, None)

        return first_state_loss, latent_loss


def render_observation_in_window(window_name, observation, mean_image=None):
    drawable_state = observation
    drawable_state = drawable_state.data.cpu().numpy()

    if mean_image is not None:
        drawable_state -= mean_image

    frame_data = (drawable_state * 255.0)

    frame_data[frame_data < 0] = 0
    frame_data[frame_data > 255] = 255
    frame_data = frame_data.astype(np.uint8)

    cv2.imshow(window_name, frame_data)
    cv2.waitKey(1)


class WarpFrameGrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        #self.res = 84
        box = self.observation_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(box.shape[0], box.shape[1], 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        #frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
        #    resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((frame.shape[0], frame.shape[1], 1))


if __name__ == '__main__':
    main()