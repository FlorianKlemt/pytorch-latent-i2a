from envs import WrapPyTorch
import torch
import time
import os
from torch.autograd import Variable
import torch.nn.functional as F
from custom_envs import make_custom_env, MiniFrameStack

from multiprocessing import Process

from baselines.common.atari_wrappers import *
import collections


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, low = 0., high = 1.):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.num_frames = num_frames
        self.frames = collections.deque(maxlen=self.num_frames)
        shp = env.observation_space.shape
        shp = (num_frames * shp[0],) +shp[1:]
        self.observation_space = spaces.Box(low=low, high=high, shape=shp, dtype=np.float32)

    def reset(self):
        self.frames.clear()
        obs = self.env.reset()
        for i in range(self.num_frames):
            self.frames.append(obs)
        return self.observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(), reward, done, info

    def observation(self):
        obs = np.expand_dims(np.concatenate(self.frames, axis=0), axis=0)
        return obs

def make_test_env(env_id):
    if 'MiniPacman' in env_id:
        env = make_custom_env(env_id, 42, 1, None)()
        env = FrameStack(env, 4)
    else:
        env = gym.make(env_id)
        env = wrap_deepmind(env)
        env = WrapPyTorch(env)
    return env


class TestEnvironment():
    def __init__(self, env, model, load_path, cuda):
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.env = env
        self.model = model
        self.model.cpu()
        saved_state = torch.load(load_path, map_location=lambda storage,loc: storage)
        for key, val in saved_state.items():
            saved_state[key] = val.cpu()

        self.model.load_state_dict(saved_state)
        self.use_cuda = cuda
        if cuda:
            self.model.cuda()
        self.reset()


    def render(self):
        self.env.render()

    def step(self):
        input = Variable(torch.from_numpy(self.state).float(), volatile = True)
        if self.use_cuda:
            input = input.cuda()

        value, logits = self.model(input)
        probs = F.softmax(logits, dim=0)
        action = probs.multinomial().data
        cpu_actions = action.cpu()
        cpu_actions = cpu_actions.numpy()
        cpu_actions = cpu_actions[0][0]

        state, reward, done, info = self.env.step(cpu_actions)
        self.reward += reward

        return done

    def reset(self):
        self.state = self.env.reset()
        self.reward = 0

    def play_game(self):
        self.reset()
        while (self.step() == False):
            self.render()
            time.sleep(0.2)
        self.render()
        return self.reward


def test_policy(env, model_type, load_path, cuda):
    env = make_test_env(env_id=env)
    i = 1
    # TODO
    while(True):
        print("started game", i)
        test = TestEnvironment(env, model_type, load_path, cuda)
        reward = test.play_game()
        print("finished game", i, ": reward", reward)
        time.sleep(10)
        i += 1


class TestPolicy():
    def __init__(self, env_id, model, load_path, cuda):
        self.model = model
        self.load_path = os.path.join(load_path, env_id + ".pt")
        self.cuda = cuda
        self.env_id = env_id
        self.p = Process(target = test_policy,
                    args=(self.env_id, self.model, self.load_path, self.cuda))
        self.p.start()