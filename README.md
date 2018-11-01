# Pytorch Latent I2A

This is a PyTorch implementation of
* Imagination Augmented Agent (I2A) [I2A](https://arxiv.org/abs/1707.06203)
* Latent Space Imagination Augmented Agent, based on [LatentI2A](https://arxiv.org/pdf/1802.03006.pdf)

To cite this repository please use the following bibtex:
```bash
@misc{repo,
  author = {Florian Klemt, Angela Denninger},
  title = {Pytorch Latent I2A},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/FlorianKlemt/pytorch-latent-i2a.git}},
  urldate = {2018-10-18}
}
```

# Requires:
* Python 3
* [PyTorch](http://pytorch.org/), tested with version 0.4.1
* [Visdom](https://github.com/facebookresearch/visdom)
* To use MiniPacman environments you also need to download and install the gym-minipacman repo [MiniPacman](https://github.com/FlorianKlemt/gym-minipacman).

In order to install requirements:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# MiniPacman
git clone https://github.com/FlorianKlemt/gym-minipacman.git
cd baselines
pip install -e .
```

# Reproducing our results:
## Train A2C on MsPacman:
```bash
--env-name MsPacmanNoFrameskip-v0 --algo a2c --train-on-200x160-pixel --entropy-coef 0.01 --num-stack 4 --num-processes 8
```
## Train Latent Space Environment Models for MsPacman:
```bash
--env-name MsPacmanNoFrameskip-v0 --environment-model dSSM_DET --lr 0.0001 --weight-decay 0 --batch-size 30 --sample-memory-size 100 --rollout-steps 10
--env-name MsPacmanNoFrameskip-v0 --environment-model dSSM_VAE --lr 0.0001 --weight-decay 0 --batch-size 15 --sample-memory-size 50 --rollout-steps 10
--env-name MsPacmanNoFrameskip-v0 --environment-model sSSM --lr 0.0001 --weight-decay 0 --batch-size 5 --sample-memory-size 20 --rollout-steps 10
```

## Train Latent Space I2A on MsPacman:
```bash
--env-name MsPacmanNoFrameskip-v0 --algo i2a --distill-coef 10 --entropy-coef 0.01 --num-stack 4 --num-processes 8 --environment-model dSSM_DET
```

## Train Classic I2A on MiniPacman Hunt:
```bash
--env-name HuntMiniPacmanNoFrameskip-v0 --algo i2a --environment-model MiniModel --log-interval 10 --num-processes 64 --num-stack 1 --distill-coef 10 --entropy-coef 0.02
```

## Train Classic I2A Copy Model on MiniPacman Hunt:
```bash
--environment-model CopyModel --env-name HuntMiniPacmanNoFrameskip-v0 --algo i2a --log-interval 10 --num-processes 32 --num-stack 1 --distill-coef 10 --entropy-coef 0.02
```

## Train A2C on MiniPacman Hunt:
```bash
--env-name HuntMiniPacmanNoFrameskip-v0 --algo a2c --entropy-coef 0.02 --num-stack 1 --num-processes 32
```

# TODO: visdom server

# Use pretrained models
TODO

# Results
TODO
