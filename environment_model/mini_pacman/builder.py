
def get_save_environment_model_path(args):
    from environment_model.model_saver import save_environment_model_path
    return save_environment_model_path(args.save_environment_model_dir,
                                       args.env_name,
                                       args.environment_model == "MiniModelLabels",
                                       args.grey_scale)
def get_log_path(args):
    from environment_model.model_saver import save_environment_model_log_path
    return save_environment_model_log_path(args.save_environment_model_dir,
                                           args.env_name,
                                           args.environment_model == "MiniModelLabels",
                                           args.grey_scale)

class MiniPacmanEnvironmentBuilder():
    def __init__(self, args):
        self.save_environment_model_path = get_save_environment_model_path(args)
        self.args = args
        self.env = self.build_env()
        self.reward_bins = self.env.unwrapped.reward_bins

    def _build_mini_pacman_copy_environment_model(self):
        from environment_model.mini_pacman.model.env_copy_model import CopyEnvModel
        return CopyEnvModel()

    def _build_mini_pacman_class_labels_environment_model(self, env):
        from environment_model.mini_pacman.model.env_model_label import MiniPacmanEnvModelClassLabels
        labels = 7
        em_obs_shape = (labels, env.observation_space.shape[1], env.observation_space.shape[2])
        return MiniPacmanEnvModelClassLabels(obs_shape=em_obs_shape,
                                             num_actions=env.action_space.n,
                                             reward_bins=self.reward_bins,
                                             use_cuda=self.args.cuda)

    def _build_mini_pacman_environment_model(self, env):
        from environment_model.mini_pacman.model.env_model import MiniPacmanEnvModel
        return MiniPacmanEnvModel(obs_shape=env.observation_space.shape,
                                  num_actions=env.action_space.n,
                                  reward_bins=self.reward_bins,
                                  use_cuda=self.args.cuda)

    def build_env(self):
        from gym_envs.envs_mini_pacman import make_custom_env
        env = make_custom_env(self.args.env_name,
                              seed=1,
                              rank=1,
                              log_dir=None,
                              grey_scale=self.args.grey_scale)()
        return env



    def build_environment_model(self, env):

        if self.args.environment_model == "CopyModel":
            environment_model = self._build_mini_pacman_copy_environment_model()
        else:
            if self.args.environment_model == "MiniModelLabels":
                environment_model = self._build_mini_pacman_class_labels_environment_model(env)
            else:
                environment_model = self._build_mini_pacman_environment_model(env)

            if self.args.load_environment_model:
                import torch
                print("Load environment model", self.save_environment_model_path)
                saved_state = torch.load(self.save_environment_model_path,
                                         map_location=lambda storage, loc: storage)
                environment_model.load_state_dict(saved_state)
            else:
                print("Save environment model under", self.save_environment_model_path)

        if self.args.cuda:
            environment_model.cuda()

        return environment_model

    def build_policy(self, env):
        from i2a.mini_pacman.i2a_mini_model import I2A_MiniModel
        from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
        policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=env.observation_space.shape,
                                                 action_space=env.action_space.n,
                                                 use_cuda=self.args.cuda))
        if self.args.cuda:
            policy.cuda()

        if not self.args.no_policy_model_loading:
            import torch
            load_policy_model_path = '{0}{1}.pt'.format(self.args.load_policy_model_dir, self.args.env_name)
            saved_state = torch.load(load_policy_model_path, map_location=lambda storage, loc: storage)
            print("Load Policy Model", load_policy_model_path)
            policy.load_state_dict(saved_state)
        return policy

    def build_loss_printer(self):
        if self.args.vis:
            from visdom import Visdom
            viz = Visdom(port=self.args.port)
        else:
            viz = None

        log_path = get_log_path(self.args)

        from environment_model.visualizer.env_mini_pacman_logger import LoggingMiniPacmanEnvTraining
        loss_printer = LoggingMiniPacmanEnvTraining(log_name=log_path,
                                                    batch_size=self.args.batch_size,
                                                    delete_log_file=self.args.load_environment_model == False,
                                                    viz=viz)
        return loss_printer

    def build_optimizer(self, environment_model):
        if self.args.environment_model == "MiniModelLabels":
            from environment_model.mini_pacman.optimizers.env_optimizer_label import EnvMiniPacmanLabelsOptimizer
            optimizer_type = EnvMiniPacmanLabelsOptimizer
        else:
            from environment_model.mini_pacman.optimizers.env_optimizer import EnvMiniPacmanOptimizer
            optimizer_type = EnvMiniPacmanOptimizer

        optimizer = optimizer_type(model=environment_model,
                                   reward_loss_coef=self.args.reward_loss_coef,
                                   lr=self.args.lr,
                                   eps=self.args.eps,
                                   weight_decay=self.args.weight_decay,
                                   use_cuda=self.args.cuda)
        return optimizer

    def build_environment_model_trainer(self, env, policy, environment_model):
        # Training Data Creator
        from environment_model.mini_pacman.training_data_creator import TrainingDataCreator
        data_creator = TrainingDataCreator(env=env,
                                           policy=policy,
                                           use_cuda=self.args.cuda)
        # Model Saver
        from environment_model.model_saver import ModelSaver
        model_saver = ModelSaver(save_model_path=self.save_environment_model_path,
                                 save_interval=self.args.save_interval)

        # Loss Printer
        loss_printer = self.build_loss_printer()

        optimizer = self.build_optimizer(environment_model=environment_model)

        from environment_model.env_model_trainer import EnvironmentModelTrainer
        trainer = EnvironmentModelTrainer(optimizer=optimizer,
                                          training_data_creator=data_creator,
                                          model_saver=model_saver,
                                          loss_printer=loss_printer,
                                          use_cuda=self.args.cuda)
        return trainer

    def build_environment_model_tester(self, env, policy, environment_model):
        from rl_visualization.environment_model.test_environment_model import TestEnvironmentModelMiniPacman
        import copy
        test_process = TestEnvironmentModelMiniPacman(env=env,
                                                      environment_model=copy.deepcopy(environment_model),
                                                      load_path=self.save_environment_model_path,
                                                      rollout_policy=policy,
                                                      args=self.args)
        return test_process


    def build_rollout_policy(self, obs_shape, action_space, use_cuda):
        from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
        from i2a.mini_pacman.i2a_mini_model import I2A_MiniModel
        rollout_policy = A2C_PolicyWrapper(I2A_MiniModel(
            obs_shape=obs_shape, action_space=action_space, use_cuda=use_cuda))
        if use_cuda:
            rollout_policy.cuda()
        return rollout_policy

    def build_a2c_model(self, env):
        from i2a.mini_pacman.i2a_mini_model import I2A_MiniModel
        from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
        return A2C_PolicyWrapper(
            I2A_MiniModel(obs_shape=env.observation_space.shape,
                          action_space=env.action_space.n,
                          use_cuda=self.args.cuda))

    def build_i2a_model(self,
                        env,
                        args):
        obs_shape = (env.observation_space.shape[0] * args.num_stack, *env.observation_space.shape[1:])
        action_space = env.action_space.n

        env_model = self.build_environment_model(env)
        for param in env_model.parameters():
            param.requires_grad = False
        env_model.eval()

        rollout_policy = self.build_rollout_policy(obs_shape, action_space, args.cuda)
        for param in rollout_policy.parameters():
            param.requires_grad = True
        rollout_policy.train()

        from i2a.mini_pacman.imagination_core import ImaginationCore
        imagination_core = ImaginationCore(env_model=env_model, rollout_policy=rollout_policy,
                                           grey_scale=args.grey_scale, frame_stack=args.num_stack)



        i2a = self.createClassicI2a(action_space, args,
                                    imagination_core,
                                    obs_shape)

        from i2a.i2a_policy_wrapper import ClassicI2A_PolicyWrapper
        i2a_model = ClassicI2A_PolicyWrapper(policy=i2a, rollout_policy=rollout_policy)

        return i2a_model

    def createClassicI2a(self, action_space, args, imagination_core, obs_shape):
        from i2a.mini_pacman.models.model_free_network import ModelFreeNetwork
        model_free_network = ModelFreeNetwork(obs_shape=obs_shape,
                                              num_outputs=512)

        from i2a.mini_pacman.models.model_based_network import ModelBasedNetwork
        model_based_network = ModelBasedNetwork(number_actions=action_space,
                                                obs_shape=obs_shape,
                                                imagination_core=imagination_core,
                                                number_lstm_cells=256,
                                                rollout_steps=args.i2a_rollout_steps,
                                                use_cuda=args.cuda)

        from i2a.i2a_agent import I2A
        i2a = I2A(model_free_network=model_free_network,
                  model_based_network=model_based_network,
                  action_space=action_space)
        return i2a

