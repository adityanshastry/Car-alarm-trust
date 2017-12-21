import sys

import numpy as np
from joblib import Parallel, delayed
from common import Trainer, Utils, Constants


def main(args):
    total_trials, num_episodes = 1, 30
    lr, epsilon, gamma = 0.01, 0.01, 1.0

    train_option = int(args[0])

    if train_option == 1:
        num_jobs, alarm_consistency = int(args[1]), bool(int(args[2]))
        print "Running Sarsa with %s alarm consistency, and %d jobs" % (str(alarm_consistency), num_jobs)
        trial_ranges = Utils.get_trial_splits(total_trials, int(total_trials / num_jobs))
        sarsa_results = Parallel(n_jobs=num_jobs)(
            delayed(Trainer.train_sarsa)(num_trials=trial_range, num_episodes=num_episodes, lr=lr,
                                         gamma=gamma, total_trials=total_trials, alarm_consistency=alarm_consistency,
                                         epsilon=epsilon)
            for trial_range in trial_ranges)
        np.save(Constants.project_root + "/results/sarsa_rewards_" + str(alarm_consistency),
                np.sum(sarsa_results, axis=0))

    elif train_option == 2:
        num_jobs, alarm_consistency = int(args[1]), bool(int(args[2]))
        print "Running Q-Learning with %s alarm consistency, and %d jobs" % (str(alarm_consistency), num_jobs)
        trial_ranges = Utils.get_trial_splits(total_trials, int(total_trials / num_jobs))
        q_learning_results = Parallel(n_jobs=num_jobs)(
            delayed(Trainer.train_sarsa)(num_trials=trial_range, num_episodes=num_episodes, lr=lr,
                                         gamma=gamma, total_trials=total_trials, alarm_consistency=alarm_consistency,
                                         epsilon=epsilon)
            for trial_range in trial_ranges)
        np.save(Constants.project_root + "/results/q_learning_rewards_" + str(alarm_consistency),
                np.sum(q_learning_results, axis=0))

    elif train_option == 3:
        result_true = np.load(Constants.project_root + "/results/q_learning_rewards_True.npy")
        result_false = np.load(Constants.project_root + "/results/q_learning_rewards_False.npy")
        avg_true = np.average(result_true, axis=0)
        avg_false = np.average(result_false, axis=0)
        stddev_true = np.std(result_true, axis=0)
        stddev_false = np.std(result_false, axis=0)
        Trainer.plot_rewards_and_episodes(avg_true, stddev_true, "Q-Learning with Consistent Alarm", "o", "b")
        Trainer.plot_rewards_and_episodes(avg_false, stddev_false, "Q-Learning with Inconsistent Alarm", "^", "r")


if __name__ == '__main__':
    main(sys.argv[1:])
