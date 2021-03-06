# PyTorch Latent I2A

This is a PyTorch implementation of
* Imagination Augmented Agent ([I2A](https://arxiv.org/abs/1707.06203)) 
* Latent Space Imagination Augmented Agent ([LatentI2A](https://arxiv.org/pdf/1802.03006.pdf))

This repository is based on a fork of the pytorch-a2c-ppo-acktr repository by Ilya Kostrikov ([https://github.com/ikostrikov/pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)).

To cite our work please use the following bibtex:
```bash
@misc{repo,
  author = {Florian Klemt, Angela Denninger, Tim Meinhardt, Laura Leal{-}Taix{\'{e}}},
  title = {PyTorch Latent I2A},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/FlorianKlemt/pytorch-latent-i2a.git}},
  urldate = {2018-10-18}
}
```

If you have any questions or suggestions write us under florian.klemt@tum.de or angela.denninger@googlemail.com.
 
# Requires
* Python 3, tested on python 3.5
* [PyTorch](http://pytorch.org/), tested with version 0.4.1
* [OpenAI Gym](https://github.com/openai/gym) with [Atari Environments](https://github.com/openai/gym#atari) enabled
* [Visdom](https://github.com/facebookresearch/visdom)
* To use MiniPacman environments you also need to download and install the gym-minipacman repo [MiniPacman](https://github.com/FlorianKlemt/gym-minipacman).

In order to install MiniPacman run:
```bash
# MiniPacman
git clone https://github.com/FlorianKlemt/gym-minipacman.git
cd gym-minipacman
pip3 install -e .
```

# Train environment and I2A models
## MiniPacman
### A2C on MiniPacman Hunt
```bash
python3 main.py --env-name HuntMiniPacmanNoFrameskip-v0 --algo a2c --num-stack 1
```

### Environment model for MiniPacman Hunt
Requires a pretrained A2C model, or the flag `--no-policy-model-loading`.
```bash
python3 main_train_environment_model.py --env-name HuntMiniPacmanNoFrameskip-v0 --environment-model MiniModelLabels --weight-decay 0
```

### I2A on MiniPacman Hunt
Requires pretrained environment model.
```bash
python3 main.py --env-name HuntMiniPacmanNoFrameskip-v0 --algo i2a --environment-model MiniModelLabels --num-stack 1 --distill-coef 10 --entropy-coef 0.02
```

### I2A with copy model on MiniPacman Hunt
The copy model has the same number of weights as the I2A model, but does not imagine the future. Therefore it does not need an environment model.
```bash
python3 main.py --environment-model CopyModel --env-name HuntMiniPacmanNoFrameskip-v0 --algo i2a --num-stack 1 --distill-coef 10 --entropy-coef 0.02
```

## MsPacman
### A2C on MsPacman
```bash
python3 main.py --env-name MsPacmanNoFrameskip-v0 --algo a2c --train-on-200x160-pixel --num-stack 4
```
### Latent space environment models for MsPacman
Requires a pretrained A2C model, or the flag `--no-policy-model-loading`.
```bash
python3 main_train_environment_model.py --env-name MsPacmanNoFrameskip-v0 --environment-model dSSM_DET --lr 0.0001 --weight-decay 0 --rollout-steps 10
python3 main_train_environment_model.py --env-name MsPacmanNoFrameskip-v0 --environment-model dSSM_VAE --lr 0.0001 --weight-decay 0 --rollout-steps 10
python3 main_train_environment_model.py --env-name MsPacmanNoFrameskip-v0 --environment-model sSSM --lr 0.0001 --weight-decay 0 --rollout-steps 10
```

### LatentI2A on MsPacman
Requires a pretrained latent space environment model.
```bash
python3 main.py --env-name MsPacmanNoFrameskip-v0 --algo i2a --distill-coef 10 --entropy-coef 0.01 --num-stack 4 --environment-model dSSM_DET
```

# Visdom server
To see a visualization of the training curves during training, start a visdom server via
```bash
python3 -m visdom.server -p 8097
```
The default port used both by visdom and our code is 8097.

# Train or play pretrained models
To continue training on a pretrained model use the `--load-model` flag. The model must lie in the folder specified via the `--save-dir` flag (default: `./trained_models/`). I2A models must lie under the subfolder `./trained_models/i2a/`, A2C models must lie under the subfolder './trained_models/a2c/'. The file must be named the same as the environment name with file-ending `.pt`.

Example:
```bash
python3 main.py --env-name MsPacmanNoFrameskip-v0 --algo i2a --distill-coef 10 --num-stack 4 --environment-model dSSM_DET --load-model
```
loads the model under `./trained_models/i2a/MsPacmanNoFrameskip-v0.pt`. The `--algo`, `--num-stack` and `--environment-model` arguments must be the same as used in the loaded model.

To play with a pretrained model without continuing to train use the `--no-training` flag.

Example:
```bash
python3 main.py --env-name MsPacmanNoFrameskip-v0 --algo i2a --num-stack 4 --environment-model dSSM_DET --no-training
```

# Results
### I2A with a dSSM-DET model on MsPacmanNoFrameskip-v0
![MsPacmanNoFrameskip-v0-I2A](readme_imgs/MsPacman_mean_median_reward.png)

### Hunt MiniPacman (HuntMiniPacmanNoFrameskip-v0)
![Hunt-MiniPacman](readme_imgs/hunt_rewards_compare.png)

### Regular MiniPacman (RegularMiniPacmanNoFrameskip-v0)
![Regular-MiniPacman](readme_imgs/regular_rewards_compare.png)

