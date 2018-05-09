import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
from Autoencoder_Tests.autoencoder import plot_smooth_loss, render_observation_in_window
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

    latent_space = 256

    env, auto_encoder_model, policy = init_autoencoder_training("MsPacman-v0",#env_name="PongDeterministic-v4",
                                                                policy_model_alias=ActorCritic,
                                                                policy_model_input_channels=4,
                                                                input_size=(84,84),
                                                                latent_space=latent_space,
                                                                use_cuda=use_cuda)

    #env, auto_encoder_model, policy = init_autoencoder_training(env_name="RegularMiniPacmanNoFrameskip-v0",
    #                                                            #root_path="/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c/",
    #                                                            #policy_model_name="RegularMiniPacmanNoFrameskip-v0.pt",
    #                                                            #load_policy_model_dir="trained_models/a2c/",
    #                                                            policy_model_alias=MiniModel,
    #                                                            policy_model_input_channels=4,
    #                                                            input_size=(19,19),
    #                                                            use_cuda=use_cuda)

    env_encoder_model = EnvEncoderModel(num_inputs=1, latent_space=latent_space, encoder_space=128, action_broadcast_size=50, use_cuda=use_cuda)
    if use_cuda:
        env_encoder_model.cuda()
    loss_criterion = torch.nn.MSELoss()
    auto_optimizer = torch.optim.RMSprop(auto_encoder_model.parameters(), lr=0.00005, weight_decay=1e-5)             #0.0001!!!
    env_encoder_optimizer = torch.optim.RMSprop(env_encoder_model.parameters(), lr=0.00005, weight_decay=1e-5)       #0.0001!!!

    latent_space_trainer = LatentSpaceEnvModelTrainer(auto_encoder_model, env_encoder_model, loss_criterion, auto_optimizer, env_encoder_optimizer, use_cuda)
    train_env_encoder(env, policy, latent_space_trainer, use_cuda)



def init_autoencoder_training(env_name, policy_model_alias, policy_model_input_channels, input_size, latent_space, use_cuda):
    from Autoencoder_Tests.AutoEncoderModel import AutoEncoderModel
    from minipacman_envs import make_minipacman_env_no_log
    #create environment to train on
    if "MiniPacman" in env_name:
        env = make_minipacman_env_no_log(env_name)
    else:
        from envs import WrapPyTorch
        from baselines.common.atari_wrappers import wrap_deepmind
        env = WrapPyTorch(wrap_deepmind(gym.make(env_name)))

    action_space = env.action_space.n

    policy = policy_model_alias(num_inputs=policy_model_input_channels, action_space=action_space, use_cuda=use_cuda)
    if use_cuda:
        policy.cuda()

    encoder_model = AutoEncoderModel(num_inputs = 1, input_size=input_size, latent_space=latent_space, hidden_space=512)
    if use_cuda:
        encoder_model.cuda()
    return env, encoder_model, policy



def train_env_encoder(env, policy, latent_space_trainer, use_cuda):
    auto_loss_plot, latent_pred_loss_plot = init_loss_plot()

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    chance_of_random_action = 0.25
    auto_loss_list = []
    latent_pred_loss_list = []
    for i_episode in range(10000):
        state = env.reset()

        done = False
        game_step_counter = 0
        total_game_auto_loss = 0
        total_game_pred_loss = 0

        while not done:
            game_step_counter += 1

            first_state_variable = Variable(torch.from_numpy(state).unsqueeze(0).type(FloatTensor))

            # let policy decide on next action
            critic, actor = policy(first_state_variable)

            prob = F.softmax(actor, dim=1)
            action = prob.multinomial().data
            action = action[0][0]
            if random.random() < chance_of_random_action:
                action = random.randint(0, env.action_space.n - 1)

            # perform action to get next state
            next_state, reward, done, _ = env.step(action)

            second_state_variable = Variable(torch.from_numpy(next_state).unsqueeze(0).type(FloatTensor))
            first_state_loss, latent_loss = latent_space_trainer.train_env_model_step(first_state_variable=first_state_variable, second_state_variable=second_state_variable, action=action)
            total_game_auto_loss += first_state_loss.data[0]
            total_game_pred_loss += latent_loss.data[0]
            state = next_state


        print("Episode ", i_episode, " auto_loss: ", total_game_auto_loss / game_step_counter, " pred_loss: ",
              total_game_pred_loss / game_step_counter)
        auto_loss_list.append(first_state_loss.data[0])
        latent_pred_loss_list.append(latent_loss.data[0])
        plot_smooth_loss(i_episode, auto_loss_list, auto_loss_plot)
        plot_smooth_loss(i_episode, latent_pred_loss_list, latent_pred_loss_plot)



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


def init_loss_plot():
    fig = plt.figure()

    auto_loss_plot = plt.subplot(121)
    auto_loss_plot.set_yscale('log')     #gca for get_current_axis
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

if __name__ == '__main__':
    main()