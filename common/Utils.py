from __future__ import division
import numpy as np
from sklearn.utils.extmath import cartesian

import Constants


def scale_to_fourier_basis(value, bounds):
    return (value - bounds[0]) / (bounds[1] - bounds[0])


def update_states_to_bounds(state):
    state[0] = max(state[0], Constants.states[0][0])
    state[0] = min(state[0], Constants.states[0][1])

    state[1] = max(state[1], Constants.states[1][0])
    state[1] = min(state[1], Constants.states[1][1])

    return state


def get_fourier_basis_constants(fourier_basis_order):
    return cartesian([np.arange(0, fourier_basis_order+1, 1), np.arange(0, fourier_basis_order+1, 1)])


def get_action_distribution(max_action, num_actions, epsilon):
    action_distribution = np.ones(shape=num_actions) * epsilon / num_actions
    action_distribution[Constants.actions[max_action]] = 1 - epsilon + (epsilon / num_actions)
    return action_distribution


def get_trial_splits(max_trials):
    starts = range(0, max_trials, 100)
    ranges = []
    for index, start in enumerate(starts):
        if index < len(starts):
            ranges.append([start, start+100])
    return ranges
    pass


def get_probabilities_for_observations(observations_df):
    observation_stats = {}

    total_instances = len(observations_df.index)

    observation_stats["age"] = {}
    observation_stats["age"][0] = observations_df.age[observations_df["age"] == 0].count() / total_instances
    observation_stats["age"][1] = observations_df.age[observations_df["age"] == 1].count() / total_instances
    observation_stats["age"][2] = observations_df.age[observations_df["age"] == 2].count() / total_instances
    observation_stats["age"][3] = observations_df.age[observations_df["age"] == 3].count() / total_instances

    observation_stats["accidents"] = {}
    observation_stats["accidents"][0] = observations_df.accidents[observations_df["accidents"] == 0].count() / total_instances
    observation_stats["accidents"][1] = observations_df.accidents[observations_df["accidents"] == 1].count() / total_instances
    observation_stats["accidents"][2] = observations_df.accidents[observations_df["accidents"] == 2].count() / total_instances

    observation_stats["fatalities"] = {}
    observation_stats["fatalities"][0] = observations_df.fatalities[observations_df["fatalities"] == 0].count() / total_instances
    observation_stats["fatalities"][1] = observations_df.fatalities[observations_df["fatalities"] == 1].count() / total_instances
    observation_stats["fatalities"][2] = observations_df.fatalities[observations_df["fatalities"] == 2].count() / total_instances

    observation_stats["sex"] = {}
    observation_stats["sex"][1] = observations_df.sex[observations_df["sex"] == 1].count() / total_instances
    observation_stats["sex"][2] = observations_df.sex[observations_df["sex"] == 2].count() / total_instances

    observation_stats["alcohol"] = {}
    observation_stats["alcohol"][1] = observations_df.alcohol[observations_df["alcohol"] == True].count() / total_instances
    observation_stats["alcohol"][0] = observations_df.alcohol[observations_df["alcohol"] == False].count() / total_instances

    observation_stats["drugs"] = {}
    observation_stats["drugs"][1] = observations_df.drugs[observations_df["drugs"] == True].count() / total_instances
    observation_stats["drugs"][0] = observations_df.drugs[observations_df["drugs"] == False].count() / total_instances

    observation_stats["distracted"] = {}
    observation_stats["distracted"][1] = observations_df.distracted[
                                             observations_df["distracted"] == True].count() / total_instances
    observation_stats["distracted"][0] = observations_df.distracted[
                                             observations_df["distracted"] == False].count() / total_instances

    return observation_stats


def main():
    print get_trial_splits(100)


if __name__ == '__main__':
    main()
