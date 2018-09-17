import torch

def save_model_name(env_name,
                    use_class_labels = False,
                    grey_scale = False):
    color_prefix = 'grey_scale' if grey_scale else 'RGB'
    class_labels_prefix = '_labels' if use_class_labels else ''
    model_name = '{0}{1}{2}'.format(env_name, color_prefix, class_labels_prefix)
    return model_name

def save_environment_model_path(save_environment_model_dir,
                                env_name,
                                use_class_labels = False,
                                grey_scale = False):
    model_name = save_model_name(env_name= env_name,
                                 use_class_labels=use_class_labels,
                                 grey_scale= grey_scale)
    return '{0}{1}.dat'.format(save_environment_model_dir, model_name)

def save_environment_model_log_path(save_environment_model_dir,
                                    env_name,
                                    use_class_labels = False,
                                    grey_scale = False):
    model_name = save_model_name(env_name = env_name,
                                 use_class_labels=use_class_labels,
                                 grey_scale=grey_scale)
    log_path = '{0}env_{1}.log'.format(save_environment_model_dir, model_name)
    return log_path


class ModelSaver():
    def __init__(self,
                 save_model_path,
                 save_interval):
        self.save_model_path = save_model_path
        self.save_interval = save_interval

    def save(self, model, episode):
        if (episode % self.save_interval) == 0:
            print("Save model", self.save_model_path)
            state_to_save = model.state_dict()
            torch.save(state_to_save, self.save_model_path)

