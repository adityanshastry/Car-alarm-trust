from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import Constants
from model import Agent, Environment


def plot_rewards_and_episodes(rewards, stddev, control_type, line_type, color):
    y_axis_ticks = np.arange(-1000, 1, 100)
    x_axis_ticks = np.arange(0, 201, 50)
    fig, ax = plt.subplots()
    ax.set_yticks(y_axis_ticks)
    ax.set_xticks(x_axis_ticks)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.errorbar(range(len(rewards)), rewards, stddev, marker=line_type, color=color, ecolor="g")
    plt.axis([0, len(rewards), -1000, 0])
    plt.title("Undiscounted Returns vs Episodes for " + control_type)
    plt.xlabel("Episodes")
    plt.ylabel("Undiscounted Returns")
    plt.show()


def train_sarsa(observations_file, log_crash_prior, alarm_consistency, num_trials, num_episodes, lr, epsilon, gamma,
                total_trials):
    undiscounted_returns = np.zeros(shape=(total_trials, num_episodes))

    environment = Environment.Environment(observations_file, log_crash_prior, alarm_consistency)
    agent = Agent.Agent(epsilon=epsilon)

    for trial in xrange(num_trials[0], num_trials[1]):
        print "Trial: ", trial
        agent.reset()
        for episode in xrange(num_episodes):
            # print "Episode: ", episode
            environment.reset()
            current_state = environment.current_state
            current_action = agent.get_action(current_state)
            total_undiscounted_reward = 0
            while False or not environment.done:
                next_state, reward, done = environment.step(current_action)
                next_action = agent.get_action(next_state)
                agent.sarsa_update(current_state, current_action, reward, next_state, next_action, lr, gamma)
                current_state = next_state
                current_action = next_action
                total_undiscounted_reward += reward

            undiscounted_returns[trial][episode] = total_undiscounted_reward

    return undiscounted_returns


def train_q_learning(observations_file, log_crash_prior, alarm_consistency, num_trials, num_episodes, lr, epsilon,
                     gamma, total_trials):
    undiscounted_returns = np.zeros(shape=(total_trials, num_episodes))

    environment = Environment.Environment(observations_file, log_crash_prior, alarm_consistency)
    agent = Agent.Agent(epsilon=epsilon)
    for trial in xrange(num_trials[0], num_trials[1]):
        print "Trial: ", trial
        agent.reset()
        for episode in xrange(num_episodes):
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
