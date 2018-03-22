from Environment_Model.environment_model_trainer import train_em
from Environment_Model.environment_model import EMModel_LSTM_One_Reward
from Environment_Model.environment_model import EMModel_used_for_Pong_I2A
from Environment_Model.environment_model import EMModel_Same_Kernel_Size
from Environment_Model.environment_model import PongEM_Big_Model

import os
import sys

root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

# small model which only predicts one reward
EMModel = EMModel_LSTM_One_Reward

# TODO ArgsParser

# Pong
train_em(atari_env="PongDeterministic-v4",
         EMModel=EMModel,
         policy_model="PongDeterministic-v4_21",
         load_policy_model_dir="trained_models/",
         environment_model_name="pong_em_one_reward",
         save_environment_model_dir="trained_models/environment_models/",
         load_environment_model=True,
         load_environment_model_dir="trained_models/environment_models/",
         root_path=root_dir,
         use_cuda=True)
