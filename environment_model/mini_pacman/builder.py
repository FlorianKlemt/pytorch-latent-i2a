def get_save_model_path(args):
    from environment_model.model_saver import save_environment_model_path
    return save_environment_model_path(args.save_environment_model_dir,
                                       args.env_name,
                                       args.use_class_labels,
                                       args.grey_scale)
def get_log_path(args):
    from environment_model.model_saver import save_environment_model_log_path
    return save_environment_model_log_path(args.save_environment_model_dir,
                                           args.env_name,
                                           args.use_class_labels,
                                           args.grey_scale)

class MiniPacmanEnvironmentBuilder():
    def __init__(self, args):
        self.save_model_path = get_save_model_path(args)
        self.args = args

    def build_env(self):
        from gym_envs.envs_mini_pacman import make_custom_env
        env = make_custom_env(self.args.env_name,
                              seed=1,
                              rank=1,
                              log_dir=None,
                              grey_scale=self.args.grey_scale)()
        return env

    def build_environment_model(self, env):
        if self.args.use_class_labels:
            from environment_model.mini_pacman.model.env_model_label import MiniPacmanEnvModelClassLabels
            EMModel = MiniPacmanEnvModelClassLabels
            labels = 7
            em_obs_shape = (labels, env.observation_space.shape[1], env.observation_space.shape[2])
        else:
            from environment_model.mini_pacman.model.env_model import MiniPacmanEnvModel
            EMModel = MiniPacmanEnvModel
            em_obs_shape = env.observation_space.shape

        reward_bins = env.unwrapped.reward_bins  # [0., 1., 2., 5., 0.] for regular

        environment_model = EMModel(obs_shape=em_obs_shape,  # env.observation_space.shape,  # 4
                                    num_actions=env.action_space.n,
                                    reward_bins=reward_bins,
                                    use_cuda=self.args.cuda)

        if self.args.load_environment_model:
            import torch
            print("Load environment model", self.save_model_path)
            saved_state = torch.load(self.save_model_path,
                                     map_location=lambda storage, loc: storage)
            environment_model.load_state_dict(saved_state)
        else:
            print("Save environment model under", self.save_model_path)

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
        if self.args.use_class_labels:
            from environment_model.mini_pacman.env_optimizer_label import EnvMiniPacmanLabelsOptimizer
            optimizer_type = EnvMiniPacmanLabelsOptimizer
        else:
            from environment_model.mini_pacman.env_optimizer import EnvMiniPacmanOptimizer
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
        model_saver = ModelSaver(save_model_path=self.save_model_path,
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

    def build_tester(self, env, policy, environment_model):
        from rl_visualization.environment_model.test_environment_model import TestEnvironmentModelMiniPacman
        import copy
        test_process = TestEnvironmentModelMiniPacman(env=env,
                                                      environment_model=copy.deepcopy(environment_model),
                                                      load_path=self.save_model_path,
                                                      rollout_policy=policy,
                                                      args=self.args)
        return test_process
