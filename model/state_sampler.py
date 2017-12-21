from common import Utils, Constants
import numpy as np
import pandas as pd
from sklearn.utils.extmath import cartesian


def get_sample_with_probability(probabilities):

    sample_with_probability = dict()
    sample_with_probability["value"] = np.random.choice(a=probabilities.keys(), p=probabilities.values())
    sample_with_probability["probability"] = probabilities[sample_with_probability["value"]]

    return sample_with_probability


def sample_observations(observations_df):
    obs_probabilities = Utils.get_probabilities_for_observations(observations_df)

    sampled_observations = dict()

    for state_key in Constants.sample_state_keys:
        sampled_observations[state_key] = dict()
        sampled_observations[state_key] = get_sample_with_probability(obs_probabilities[state_key])

    return sampled_observations

    pass


def get_state_with_crash_time(observations_file, log_crash_prior):
    observations_df = pd.read_csv(observations_file, sep=",")

    sampled_observations = sample_observations(observations_df)
    # state = []

    crash_posterior = log_crash_prior
    for state_key in Constants.sample_state_keys:
        # state.append(sampled_observations[state_key]["value"])
        crash_posterior += np.log(sampled_observations[state_key]["probability"])

    # state.extend(Constants.default_crash_alarm_values)

    return Constants.default_crash_alarm_values[0], Constants.default_crash_alarm_values[1], -int(crash_posterior)


if __name__ == '__main__':
    processed_obs_file = "../data/processed_data.csv"
    print get_state_with_crash_time(processed_obs_file, Constants.log_crash_prior)
