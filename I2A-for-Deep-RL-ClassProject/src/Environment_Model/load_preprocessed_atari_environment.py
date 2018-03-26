from environment import atari_env
from utils import read_config

def load_atari_environment(atari_game_name):
    class ar:
        def __init__(self):
            self.skip_rate = 4

    setup_json = read_config('config.json')
    env_config = setup_json["Default"]
    for i in setup_json.keys():
        if i in atari_game_name:
            env_config = setup_json[i]

    args = ar()
    env = atari_env(atari_game_name, env_config, args)
    return env