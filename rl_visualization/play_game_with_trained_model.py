from envs import WrapPyTorch
import torch
import time
import os
import torch.nn.functional as F

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

def make_test_env(env_id, grey_scale, frame_stack):
    if 'MiniPacman' in env_id:
        from gym_envs.envs_mini_pacman import make_custom_env
        env = make_custom_env(env_id, 42, 1, None, grey_scale=grey_scale)()
        env = FrameStack(env, frame_stack)
    elif 'MsPacman' in env_id:
        from gym_envs.envs_ms_pacman import make_env_ms_pacman
        env = make_env_ms_pacman(env_id=env_id, seed = 42, rank=1, log_dir=None, grey_scale=False, stack_frames=frame_stack, skip_frames=4)()
    else:
        env = gym.make(env_id)
        env = wrap_deepmind(env)
        env = WrapPyTorch(env)
    return env


class TestEnvironment():
    def __init__(self, env, model, load_path, args):
        self.args = args
        self.FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.env = env
        self.model = model
        saved_state = torch.load(load_path, map_location=lambda storage,loc: storage)

        self.model.load_state_dict(saved_state)
        self.use_cuda = args.cuda
        if args.cuda:
            self.model.cuda()
        self.reset()


    def render(self):
        self.env.render()

    def step(self):
        with torch.no_grad():
            input = torch.from_numpy(self.state).float()
            if len(input.shape)==3: #if batch-dimension is missing
                input = input.unsqueeze(0)
            if self.use_cuda:
                input = input.cuda()

            value, actor = self.model(input)
            probs = F.softmax(actor, dim=1)
            action = probs.multinomial(num_samples=1)
            action = action.item()

            self.state, reward, done, info = self.env.step(action)
            self.reward += reward

        return done

    def reset(self):
        self.state = self.env.reset()
        self.reward = 0

    def play_game(self):
        self.reset()
        while (self.step() == False):
            self.render()   #this always renders RGB (the computation should be done correctly)
            time.sleep(0.05)
        self.render()
        return self.reward


def test_policy(model_type, load_path, args):
    env = make_test_env(env_id=args.env_name, grey_scale=args.grey_scale, frame_stack=args.num_stack)
    i = 1
    # TODO
    while(True):
        print("started game", i)
        test = TestEnvironment(env, model_type, load_path, args)
        reward = test.play_game()
        print("finished game", i, ": reward", reward)
        time.sleep(10)
        i += 1


class TestPolicy():
    def __init__(self, model, load_path, args):
        load_path = os.path.join(load_path, args.env_name + ".pt")
        self.p = Process(target = test_policy,
                    args=(model, load_path, args))
        self.p.start()

    def stop(self):
        self.p.terminate()