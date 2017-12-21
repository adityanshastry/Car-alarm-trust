from __future__ import division

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import Constants
from model import Agent, Environment


def plot_rewards_and_episodes(rewards, stddev, control_type, line_type, color):
    y_axis_ticks = np.arange(-2, 2, 0.5)
    x_axis_ticks = np.arange(0, 201, 50)
    fig, ax = plt.subplots()
    ax.set_yticks(y_axis_ticks)
    ax.set_xticks(x_axis_ticks)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.errorbar(range(len(rewards)), rewards, stddev, marker=line_type, color=color, ecolor="g")
    plt.title("Undiscounted Returns vs Episodes for " + control_type)
    plt.xlabel("Episodes")
    plt.ylabel("Undiscounted Returns")
    plt.show()


def train_sarsa(num_trials, num_episodes, lr, gamma, total_trials, alarm_consistency, epsilon):
    undiscounted_returns = np.zeros(shape=(total_trials, num_episodes))
    decayed_epsilon = epsilon
    environment = Environment.Environment(Constants.observations_file, Constants.log_crash_prior, alarm_consistency)
    agent = Agent.Agent(epsilon=epsilon)

    for trial in xrange(num_trials[0], num_trials[1]):
        agent.reset(epsilon)
        for episode in xrange(num_episodes):
            # if episode % Constants.epsilon_decay_episode:
            #     decayed_epsilon /= 2
            #     agent.set_epsilon(decayed_epsilon)

            print "Trial: %d, Episode: %d" % (trial, episode)
            environment.reset()
            current_state = environment.current_state
            current_action = agent.get_action(current_state)
            total_undiscounted_reward = 0
            while False or not environment.done:
                next_state, reward, done = environment.step(current_action)
                next_action = agent.get_action(next_state)
                # print current_state, current_action, reward, next_state, next_action
                agent.sarsa_update(current_state, current_action, reward, next_state, next_action, lr, gamma)
                current_state = next_state
                current_action = next_action
                total_undiscounted_reward += reward
            undiscounted_returns[trial][episode] = total_undiscounted_reward

    return undiscounted_returns


def train_q_learning(num_trials, num_episodes, lr, gamma, total_trials, alarm_consistency, epsilon):
    undiscounted_returns = np.zeros(shape=(total_trials, num_episodes))

    environment = Environment.Environment(Constants.observations_file, Constants.log_crash_prior, alarm_consistency)
    agent = Agent.Agent(epsilon=epsilon)

    for trial in xrange(num_trials[0], num_trials[1]):
        agent.reset()
        for episode in xrange(num_episodes):
            print "Trial: %d, Episode: %d" % (trial, episode)
            environment.reset()
            current_state = environment.current_state
            current_action = agent.get_action(current_state)
            total_undiscounted_reward = 0
            while False or not environment.done:
                next_state, reward, done = environment.step(current_action)
                agent.q_learning_update(current_state, current_action, reward, next_state, lr, gamma)
                current_state = next_state
                current_action = agent.get_action(next_state)
                total_undiscounted_reward += reward

            undiscounted_returns[trial][episode] = total_undiscounted_reward

    return undiscounted_returns


if __name__ == '__main__':
    # observations_file = "../data/processed_data.csv"
    # np.save("../results/sarsa",
    #         train_sarsa(observations_file, Constants.log_crash_prior, True, (0, 1), 100, 0.01, 0.01, 1, 1))
    pass
