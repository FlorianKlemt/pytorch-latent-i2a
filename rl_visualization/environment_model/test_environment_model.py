import torch
import numpy as np
import time
import cv2
from random import randint
import os


from multiprocessing import Process

class RenderMiniPacmanImaginationCore():
    def __init__(self, grey_scale):
        self.grey_scale = grey_scale

        render_window_sizes = (1520, 760)
        self.window_name = 'imagination_core'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, render_window_sizes)


    def render_preprocessing(self, observation, reward_text, step_text):
        drawable_state = observation.view(observation.shape[1], observation.shape[2], -1)

        drawable_state = drawable_state.detach().cpu().numpy()

        zeros = np.ones((4, drawable_state.shape[1], drawable_state.shape[2]))
        drawable_state = np.append(drawable_state, zeros, axis=0)
        drawable_state = drawable_state.repeat(40,0).repeat(40,1)

        frame_data = (drawable_state * 255.0)

        frame_data[frame_data < 0] = 0
        frame_data[frame_data > 255] = 255
        frame_data = frame_data.astype(np.uint8)

        cv2.putText(frame_data, reward_text, (20, 720), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(frame_data, step_text, (20, 680), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)

        if not self.grey_scale:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

        return frame_data

    def render_observation(self, observation, predicted_observation, reward, predicted_reward, rollout_step, save_path=None):
        frame_data1 = self.render_preprocessing(observation, 'true reward: {0:.3f} '.format(reward), 'rollout step: '+str(rollout_step))
        frame_data2 = self.render_preprocessing(predicted_observation, 'predicted reward: {0:.3f} '.format(predicted_reward), 'rollout step: '+str(rollout_step))

        both = np.hstack((frame_data1, frame_data2))

        cv2.imshow(self.window_name, both)
        cv2.waitKey(1000)



class RenderImaginationCore():
    def __init__(self, grey_scale):
        self.grey_scale = grey_scale

        render_window_sizes = (640, 400)
        self.window_name = 'imagination_core'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, render_window_sizes)


    def render_preprocessing(self, observation, reward_text, step_text):
        drawable_state = observation.detach().cpu().numpy()
        drawable_state = np.transpose(drawable_state, (1, 2, 0))

        frame_data = (drawable_state * 255.0)
        frame_data[frame_data < 0] = 0
        frame_data[frame_data > 255] = 255
        frame_data = frame_data.astype(np.uint8)

        if not self.grey_scale:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

        return frame_data

    def render_observation(self, observation, predicted_observation, reward, predicted_reward, rollout_step, save_path):
        frame_data1 = self.render_preprocessing(observation, 'true reward: {0:.3f} '.format(reward), 'rollout step: '+str(rollout_step))
        frame_data2 = self.render_preprocessing(predicted_observation, 'predicted reward: {0:.3f} '.format(predicted_reward), 'rollout step: '+str(rollout_step))

        if save_path is not None:
            cv2.imwrite(save_path+'obs_'+str(rollout_step)+'.png', frame_data1)
            cv2.imwrite(save_path+'pred_'+str(rollout_step)+'.png', frame_data2)

        both = np.hstack((frame_data1, frame_data2))

        cv2.imshow(self.window_name, both)
        cv2.waitKey(1000)


def numpy_to_variable(numpy_value, use_cuda):
    value = torch.from_numpy(numpy_value).unsqueeze(0).float()
    if use_cuda:
        value = value.cuda()
    return value

def play_with_imagination_core(imagination_core, renderer, env, args, game_nr):
    save_base_path = args.env_model_images_save_path
    save_path = None
    if save_base_path is not None:
        save_path = save_base_path+'Game'+str(game_nr)+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    render = True

    observation = env.reset()
    state = numpy_to_variable(observation, args.cuda)
    state_stack = torch.cat((state, state, state), 0)

    # do 20 random actions to get different start observations
    for i in range(randint(100, 200)):
        observation, reward, done, _ = env.step(env.action_space.sample())
        state = numpy_to_variable(observation, args.cuda)
        state_stack = torch.cat((state_stack, state), 0)
        state_stack = state_stack[1:]

    predicted_state = state

    if args.use_latent_space:
        latent_state = imagination_core.encode(state_stack.unsqueeze(0))
        predicted_state, predicted_reward = imagination_core.decode(latent_state, None)

    renderer.render_observation(state[0], predicted_state[0], reward, reward, 0, save_path)
    latent_state = None

    for t in range(5):
        if args.use_latent_space:
            action = np.random.choice(env.action_space.n, 1)
            action = torch.from_numpy(action).unsqueeze(0)
            if args.cuda:
                action = action.cuda()
            if latent_state is None:
                latent_state = imagination_core.encode(state_stack.unsqueeze(0))
            latent_state, z_prior, predicted_reward = imagination_core(latent_state, action)
            predicted_state, _ = imagination_core.decode(latent_state, z_prior)
        else:
            action = imagination_core.sample(predicted_state)
            predicted_state, predicted_reward = imagination_core(predicted_state, action)

        predicted_reward = predicted_reward.detach().cpu().numpy()
        observation, reward, done, _ = env.step(action.item())
        state = numpy_to_variable(observation, args.cuda)
        state_stack = torch.cat((state_stack, state), 0)
        state_stack = state_stack[1:]

        if render:
            renderer.render_observation(state[0], predicted_state[0], reward, predicted_reward.item(), t+1, save_path)



def test_environment_model(env, environment_model, load_path, rollout_policy, args):
    i = 1
    while(True):
        try:
            saved_state = torch.load(load_path, map_location=lambda storage, loc: storage)
        except:
            print("Could not load model, try again.")
            time.sleep(5)
            continue
        environment_model.load_state_dict(saved_state)

        if args.cuda:
            environment_model.cuda()

        if args.use_latent_space:
            from i2a.latent_space.latent_space_imagination_core import LatentSpaceImaginationCore
            imagination_core = LatentSpaceImaginationCore(env_model=environment_model,
                                                          rollout_policy=rollout_policy)
            renderer = RenderImaginationCore(args.grey_scale)
        else:
            from i2a.mini_pacman.imagination_core import ImaginationCore
            imagination_core = ImaginationCore(env_model=environment_model,
                                               rollout_policy=rollout_policy,
                                               grey_scale = False,
                                               frame_stack = 1)
            renderer = RenderMiniPacmanImaginationCore(args.grey_scale)

        print("started game", i)
        play_with_imagination_core(imagination_core=imagination_core, renderer=renderer, env=env, args=args, game_nr=i)
        i += 1
        time.sleep(2)




class TestEnvironmentModelMiniPacman():
    def __init__(self, env, environment_model, load_path, rollout_policy, args):
        args.use_latent_space = False
        self.p = Process(target = test_environment_model,
                         args=(env, environment_model, load_path, rollout_policy, args))
        self.p.start()

    def stop(self):
        self.p.terminate()

class TestEnvironmentModel():
    def __init__(self, env, environment_model, load_path, rollout_policy, args):
        self.p = Process(target = test_environment_model,
                         args=(env, environment_model, load_path, rollout_policy, args))
        self.p.start()

    def stop(self):
        self.p.terminate()

