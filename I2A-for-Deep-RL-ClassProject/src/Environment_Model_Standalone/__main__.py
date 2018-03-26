from agent import OverfitOnRandomFramesAgent, OverfitOnSinglePongFrameAgent, OverfitOnNFramesAgent
from models.environment_model.pongEM import PongEM
import models.ModelFreeNetwork as mfn
import torch
from utils import create_loggers
from torch.autograd import Variable

#mf_A3C = mfn.load_model_A3C('models/trained_models/A3C/Pong-v0.dat', 1, 6)
#x = Variable(torch.FloatTensor([[[[1]*80]*80]]))
#cx = Variable(torch.zeros(1, 512))
#hx = Variable(torch.zeros(1, 512))
#print(mf_A3C((x,(cx,hx))))
#exit(code=0)

# set this to get a folder name that can be recognized

experiment_name = 'IntegrateIntoA3C_PongEM400'
#experiment_name = None
from torch.autograd import Variable
experiment_name = None


experiment_name = create_loggers(experiment_name=experiment_name)


# overfit on a single frame
#agent = OverfitOnSinglePongFrameAgent()
#print("Overfitted on 1")

# overfit on random frames which are generated
# agent = OverfitOnRandomFramesAgent(1, (1, 80, 80), 6)

# overfit on 400 real pong frames.
# For this use-case you have to adjust the learning rate to smt. like 1e-6 to see something
#agent = OverfitOnNFramesAgent(dataset_name='pacman_frames_30000', num_actions=9, n_frames=1)
agent = OverfitOnNFramesAgent(dataset_name='pong_frames_400', num_actions=6, n_frames=400)

net = PongEM(name=experiment_name,
             env_state_dim=(1, 80, 80),
             num_input_actions=agent.num_game_actions,
             enable_tensorboard=False)

# train the agent with 40000 iterations
# adjust should_use_cuda if you don't have cuda

net.train(agent,
          num_iterations=100000,
          optimizer_args=net.default_adam_args,
          should_use_cuda=True,
          render=True,
          render_window_sizes=(400, 400),
          save_images_after_xth_iteration=-1,
          log_every_xth_iteration=10)

#torch.save(net,"/home/meins/Meins/Studium/DL4CV/Project/I2A-for-Deep-RL.git/trunk/I2A/emPong_400_test.model")
