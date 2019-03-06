## Introduction and objectives
This project is part of the [Udacity deep reinforcement learning nanodegree](http://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). The goal is to apply the knowledge obtained during the course to train an agent using [DDPG](https://arxiv.org/abs/1509.02971).

## Background
In the [DDPG paper](https://arxiv.org/abs/1509.02971), they introduced this algorithm as an ["Actor-Critic" method](https://cs.wmich.edu/~trenary/files/cs5300/RLBook/node66.html). Though, some researchers think DDPG is best classified as a [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) method for continuous action spaces, so it is worth it to read and understand how actor-critic methods and DQN methods works.

## Project Details
This project uses the Reacher environment.

![reacher](docs/reacher.gif "Reacher")

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The project uses 20 identical agents, each with its own copy of the environment. The task is episodic and for solving the environment the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## Getting Started
This project requires Python 3.6+, pytorch, torchvision, matplotlib and numpy to work. It's recommended to use a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/) to run the project with the appropriate requirements.

For this project, you will need to download the environment from one of the links below. You need only select the environment that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then, place the file in the root folder, and unzip (or decompress) the file.

## Instructions
To run the project navigate in your terminal to the root folder and execute `python3.6 -m continuous_control.main --environment path_to_your_environment`. For example, if you use Linux and placed the environment in the root folder following the instructions, the concrete instruction would be `python3.6 -m continuous_control.main --environment Reacher_Linux_multiple/Reacher.x86_64`. To use a trained model, include the option `--trained`: `python3.6 -m continuous_control.main --trained --environment Reacher_Linux_multiple/Reacher.x86_64`