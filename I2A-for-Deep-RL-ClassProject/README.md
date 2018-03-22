# Imagination-Augmented Agents for Deep-Reinforcement-Learning
We started this project as part of our lecture [Deep Learning for Computer Vision](https://vision.cs.tum.edu/teaching/ws2017/dl4cv) at the [TU Munich](https://www.tum.de/nc/en/homepage/). 
You can find our project proposal and our poster in the Folder [Class_Project](/Class_Project). In case you have any question feel free to get in touch with us.


As foundation for our project we used the code provided by [https://github.com/dgriff777/rl_a3c_pytorch](https://github.com/dgriff777/rl_a3c_pytorch).

## 1 Motivation and Goals
The experts from [DeepMind](https://deepmind.com/) presented the paper [Imagination-Augmented Agents for Deep Reinforcement Learning
](http://papers.nips.cc/paper/7152-imagination-augmented-agents-for-deep-reinforcement-learning) at [NIPS 2017](https://nips.cc/Conferences/2017). 
The main idea of the paper is to combine the best of model-based and model-free reinforcement learning, by providing a model, but at the same time giving the network the option to discard potentially wrong information from a non-perfect model. The model-based path uses a special rollout policy to imagine trajectories of state-reward pairs that would follow an action. Using this information the I2A net is able to prevent the agent from making non-recoverable wrong decisions.

In the I2A paper they use a custom implementations of the Atari Games Sokoban and PacMan, where their implementation of PacMan is a special Mini-Pacman with 15x19 RGB images size and their Sokoban implementation has a 80x80 RGB image as input.
Due to limited time during the class project, we decided to use [OpenAi Gym](https://github.com/openai/gym) 
to provide us the game environment. 
We implemented I2A sucessfully and found out that they use a Mini-Pacman version due to scaling problems, as the I2A architecture is computationally expensive. So we were only able to get I2A trained on Pong, but here we can't see the superiour quality of the I2A agent, compared to baslines like A3C, because Pong is easy to learn.

## 2 Architecture
In the following we explain the design of the network and our choice. 
Please note that we did not implement the complete same layers
as proposed in the paper [[1](#references)] due to the issues described in the [previous sections](#motivation-and-goals)

<p align="center">
  <img src="Doc/Full_I2A_Architecture.png?raw=true">
</p> 

The architecture consists of three main components: [Imagination Path](#imagination-path), 
[Model Free Path](#model-free-path), [Path Aggregator](#path-aggregator)
In the following you will find detailed descriptions for every component as well as their subcomponents.


### 2.1 Model-Based Path
This is the component where most of the magic is happening. The model-based path introduces an approximate model of the environment to the net.
Additionally it uses the ability to predict the future with the imagination core, to imagine the next n state-reward pairs following an initial action. The paper uses a rollout for every action a_i. The predicted state from each rollout step is encoded by a convolutional LSTM in which it is fed in reverse order. All encodings C_i will be concatenated to C_im.

#### 2.1.1 Single Rollout
During a single rollout the rollout encoder imagines the future of one specific action, where in each step it predicts the next frame and the next reward for a given state and action with the Imagination Core.
Starting with the intrinsic action of a_i of the Rollout i and the current state, the Imagination Core gets chained for n times.
The n results of the imagination future are evaluated by the encoding. 
During the encoding the results of the Imagination Future are feed in to an convolutional [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) 
starting with the the most distanced imagined future.


#### 2.1.2 Imagination Core
The task of the Imagination Core (IC) is to predict one step of the environment into the future based on a given input state.
The IC consists of two components, a Policy Net (PN) and an Environment Model (EM). 
The Policy Model is used to decide which action should be performed, which gives the direction the trajectory is developing in. 
The imagined action together with the input state then will be passed to the Environment Model, 
which then predicts the next state as well as the expected reward.

There is one exception for this chaining inside of the IC, namely at the beginning of a Rollout. 
In this case we already know the input action for the Environment Model, as it is defined for each Rollout and therefore we do not need to 
let the Policy Model decide which action should be imagined.


#### 2.1.3 Environment Model
#####(from paper)

The Environment Model (EM) is one of the most important elements in the whole I2A architecture. 

#### 2.1.4 Policy Net

#### 2.1.5 Rollout Encoder Network
Convolutional Encoder + LSTM

### 2.2 Model Free Path
Standard network of convolutional layers + one fc layer

### 3 Requirements
We used Anaconda for Python 3.6. All packages we used can be found in the file [I2A/i2a.yml](I2A/i2a.yml). 
To create an conda environment with our configuration just run
```conda env create -f i2a.yml```\
You will need to install the following packages by hand:
- **Gym** https://github.com/openai/gym
- **Universe** https://github.com/openai/universe

### 4 Next Steps

### 5 Repository Content

### References
[1] Racanière, Sébastien, et al. "Imagination-Augmented Agents for Deep Reinforcement Learning." Advances in Neural Information Processing Systems. 2017. [src](http://papers.nips.cc/paper/7152-imagination-augmented-agents-for-deep-reinforcement-learning)\
[2] Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International Conference on Machine Learning. 2016. \
[3] Leibfried, Felix, Nate Kushman, and Katja Hofmann. "A deep learning approach for joint video frame and reward prediction in atari games." arXiv preprint arXiv:1611.07078 (2016). \
[4] Brockman, Greg, et al. "Openai gym." arXiv preprint arXiv:1606.01540 (2016). 
