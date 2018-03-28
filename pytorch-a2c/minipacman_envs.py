import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import *

class WrapPyTorchMiniPacman(gym.ObservationWrapper):
    def __init__(self, env=None, image_size = 19):
        super(WrapPyTorchMiniPacman, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, image_size, image_size])

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)

class WarpMiniPacmanFrame(gym.ObservationWrapper):
    def __init__(self, env, image_resize_size = 19):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = image_resize_size
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1))


def make_minipacman_env(env_id, seed, rank, log_dir):
    def _thunk():
        episode_life = True
        clip_rewards = True
        image_resize_size = 19

        env = gym.make(env_id)

        #magic
        env.frameskip = 1
        #env.unwrapped.ale.setFloat('repeat_action_probability', 0.)

        env.seed(seed + rank)
        env = bench.Monitor(env,
                            os.path.join(log_dir,
                                         "{}.monitor.json".format(rank)))

        if episode_life:
            env = EpisodicLifeEnv(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=1)

        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = WarpMiniPacmanFrame(env, image_resize_size)
        if clip_rewards:
            env = ClipRewardEnv(env)

        env = WrapPyTorchMiniPacman(env)
        return env

    return _thunk


def make_minipacman_env_no_log(env_id):
    episode_life = True
    clip_rewards = True
    image_resize_size = 19

    env = gym.make(env_id)

    #magic
    env.frameskip = 1

    if episode_life:
        env = EpisodicLifeEnv(env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=1)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpMiniPacmanFrame(env, image_resize_size)
    if clip_rewards:
        env = ClipRewardEnv(env)

    env = WrapPyTorchMiniPacman(env)
    return env