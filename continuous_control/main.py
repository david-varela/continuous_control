import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from continuous_control.ddpg_agent import Agent

ACTOR_CHECKPOINT_PATH = str(Path() / 'continuous_control' / 'checkpoint_actor.pth')
CRITIC_CHECKPOINT_PATH = str(Path() / 'continuous_control' / 'checkpoint_critic.pth')


def not_trained_mode(agent, env, brain_name):
    print_every = 100
    scores_deque = deque(maxlen=print_every)
    scores = []
    episode = 0

    while True:
        episode += 1
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations
        agent.reset()
        score = 0
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += sum(rewards) / 20.0
            if dones[0]:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)), end='')
        if episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 30.0:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                       np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), ACTOR_CHECKPOINT_PATH)
            torch.save(agent.critic_local.state_dict(), CRITIC_CHECKPOINT_PATH)
            if episode > 100:
                break

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def trained_mode(agent, env, brain_name):
    agent.actor_local.load_state_dict(torch.load(ACTOR_CHECKPOINT_PATH))
    agent.critic_local.load_state_dict(torch.load(CRITIC_CHECKPOINT_PATH))
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state
    score = 0  # initialize the score
    while True:
        actions = agent.act(states)  # select one action per agent
        env_info = env.step(actions)[brain_name]  # send the actions to the environment
        next_states = env_info.vector_observations  # get the next states
        rewards = env_info.rewards  # get the rewards
        dones = env_info.local_done  # see if episode has finished
        score += sum(rewards) / 20.0  # update the score
        states = next_states  # roll over the states to next time step
        if dones[0]:  # exit loop if episode finished
            break
    print("Score: {}".format(score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', help='Load a trained agent', action='store_true')
    parser.add_argument('--environment', help='Path to the environment', default="Reacher_Linux_multiple/Reacher.x86_64")
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    if args.trained:
        trained_mode(agent, env, brain_name)
    else:
        not_trained_mode(agent, env, brain_name)
    env.close()


if __name__ == '__main__':
    main()
