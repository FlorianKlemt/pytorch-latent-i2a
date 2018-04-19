from envs import WrapPyTorch
import torch
import time
import os
from torch.autograd import Variable
import torch.nn.functional as F
from minipacman_envs import make_minipacman_env_no_log

from multiprocessing import Process

from baselines.common.atari_wrappers import *
import collections


def make_test_env(env_id, model_name):
    if model_name == "MiniModel":
        env = make_minipacman_env_no_log(env_id)
    else:
        env = gym.make(env_id)
        env = wrap_deepmind(env)
        env = WrapPyTorch(env)
    return env


class TestEnvironment():
    def __init__(self, env, model_type, num_stack, load_path, cuda):
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.num_stack = num_stack
        self.env = env
        self.load_model(model_type, load_path, num_stack, cuda)

        self.reset()

    def load_model(self, model_type, load_path, num_stack, cuda):
        self.model = model_type(self.env.observation_space.shape[0] * num_stack, self.env.action_space.n, cuda)
        self.model.load_state_dict(torch.load(load_path))
        if cuda:
            self.model.cuda()

    def render(self):
        self.env.render()

    def step(self):
        value, logits = self.model(Variable(torch.from_numpy(self.state).type(self.FloatTensor), volatile=True).cuda())
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


def test_policy(env, model_type, num_stack, load_path, cuda):
    model_name = model_type.__name__
    env = make_test_env(env_id=env,
                        model_name=model_name)
    i = 1
    # TODO
    while(True):
        print("started game", i)
        test = TestEnvironment(env, model_type, num_stack, load_path, cuda)
        reward = test.play_game()
        print("finished game", i, ": reward", reward)
        time.sleep(10)
        i += 1


class TestPolicy():
    def __init__(self, env_id, model_type, num_stack, load_path, cuda):
        self.model_type = model_type
        self.num_stack = num_stack
        self.load_path = os.path.join(load_path, env_id + ".pt")
        self.cuda = cuda
        self.env_id = env_id
        self.p = Process(target = test_policy,
                    args=(self.env_id, self.model_type, self.num_stack, self.load_path, self.cuda))
        self.p.start()

    def start_test_process(self):
        pass
        #self.p.start()
        #test_policy(self.env_id, self.model_type, self.num_stack, self.load_path, self.cuda)
        #p = Process(target = test_policy,
        #            args=(self.env_id, self.model_type, self.num_stack, self.load_path, self.cuda))
        #self.p.start()
        #p.join()