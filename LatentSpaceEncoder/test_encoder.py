import torch
from multiprocessing import Process
import time
from torch.autograd import Variable
from random import randint
import cv2
import numpy as np

class RenderLatentSpace():
    def __init__(self, grey_scale, is_mini_pacman):
        self.grey_scale = grey_scale
        self.is_mini_pacman = is_mini_pacman

        render_window_sizes = (1520, 760)
        self.window_name = 'latent_space_prediction'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, render_window_sizes)


    def render_preprocessing(self, observation, reward_text, step_text, is_mini_pacman=True):
        drawable_state = observation.detach().cpu().numpy()

        if not is_mini_pacman:
            drawable_state = np.transpose(drawable_state, (1, 2, 0))
        else:
            drawable_state = drawable_state.reshape(observation.shape[1], observation.shape[2], observation.shape[0])

        if is_mini_pacman:  #scale to fill the window
            drawable_state = drawable_state.repeat(40, 0).repeat(40, 1)

        frame_data = (drawable_state * 255.0)

        frame_data[frame_data < 0] = 0
        frame_data[frame_data > 255] = 255
        frame_data = frame_data.astype(np.uint8)


        if not self.grey_scale:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

        return frame_data

    def render_observation(self, observation, predicted_observation, reward, predicted_reward, rollout_step):
        frame_data1 = self.render_preprocessing(observation, 'true reward: {0:.3f} '.format(reward), 'rollout step: '+str(rollout_step), is_mini_pacman=self.is_mini_pacman)
        frame_data2 = self.render_preprocessing(predicted_observation, 'predicted reward: {0:.3f} '.format(predicted_reward), 'rollout step: '+str(rollout_step), is_mini_pacman=self.is_mini_pacman)

        both = np.hstack((frame_data1, frame_data2))

        cv2.imshow(self.window_name, both)
        cv2.waitKey(1000)

def numpy_to_torch(numpy_value, use_cuda):
    value = torch.from_numpy(numpy_value).unsqueeze(0).float()
    if use_cuda:
        value = value.cuda()
    return value

def play_with_latent_space(auto_encoder, env_encoder, rollout_policy, env, args):
    renderer = RenderLatentSpace(grey_scale=False, is_mini_pacman='MiniPacman' in args.env)

    observation = env.reset()
    state = numpy_to_torch(observation, args.cuda)

    # do 20 random actions to get different start observations
    for i in range(randint(20, 50)):
        observation, reward, done, _ = env.step(env.action_space.sample())
        state = numpy_to_torch(observation, args.cuda)

    renderer.render_observation(state[0], state[0], reward, reward, 0)

    predicted_state = state

    for t in range(5):
        _, action, _, _ = rollout_policy.act(predicted_state, None, None)
        state_latent = auto_encoder.encode(predicted_state)
        predicted_latent_state, predicted_reward = env_encoder(state_latent, action)

        predicted_reward = predicted_reward.detach().cpu().numpy()

        observation, reward, done, _ = env.step(action.item())
        state = numpy_to_torch(observation, args.cuda)

        #decode for renderer
        decoded_predicted_state = auto_encoder.decode(predicted_latent_state)
        renderer.render_observation(state[0], decoded_predicted_state[0], reward, predicted_reward.item(), t+1)

        predicted_state = decoded_predicted_state


def test_latent_space_model(env, auto_encoder, env_encoder, auto_encoder_load_path, env_encoder_load_path, rollout_policy, args):
    i = 1
    while(True):
        try:
            auto_encoder_saved_state = torch.load(auto_encoder_load_path, map_location=lambda storage, loc: storage)
            auto_encoder.load_state_dict(auto_encoder_saved_state)
            env_encoder_saved_state = torch.load(env_encoder_load_path, map_location=lambda storage, loc: storage)
            env_encoder.load_state_dict(env_encoder_saved_state)
        except:
            print("Models have not been saved yet, cannot test!")
            time.sleep(10)

        if args.cuda:
            auto_encoder.cuda()
            env_encoder.cuda()

        play_with_latent_space(auto_encoder=auto_encoder, env_encoder=env_encoder, rollout_policy=rollout_policy, env=env, args=args)

        print("finished game", i)
        time.sleep(5)
        i += 1

class TestLatentSpaceModel():
    def __init__(self, env, auto_encoder, env_encoder, auto_encoder_load_path, env_encoder_load_path, rollout_policy, args):
        self.p = Process(target = test_latent_space_model,
                         args=(env, auto_encoder, env_encoder, auto_encoder_load_path, env_encoder_load_path, rollout_policy, args))
        self.p.start()

    def stop(self):
        self.p.terminate()