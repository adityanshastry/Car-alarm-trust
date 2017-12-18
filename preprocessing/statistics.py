from __future__ import division
import pandas as pd
import cPickle as pickle


def initialize_observations_stats():
    observations_df = pd.read_csv("../data/preprocessed_data.csv", sep=",")

    print observations_df.accidents.unique()
    print observations_df.fatalities.unique()


def bin_preprocessed_data(observations_file):

    processed_data_file = "../data/processed_data.csv"

    observations_df = pd.read_csv(observations_file, sep=",")
    observations_df.age[observations_df["age"] < 18] = 0
    observations_df.age[(observations_df["age"] >= 18) & (observations_df["age"] < 25)] = 1
    observations_df.age[(observations_df["age"] >= 25) & (observations_df["age"] < 65)] = 2
    observations_df.age[observations_df["age"] >= 65] = 3

    observations_df.accidents[observations_df["accidents"] <= 10] = 0
    observations_df.accidents[(observations_df["accidents"] > 10) & (observations_df["accidents"] <= 50)] = 1
    observations_df.accidents[observations_df["accidents"] > 50] = 2

    observations_df.fatalities[observations_df["fatalities"] <= 5] = 0
    observations_df.fatalities[(observations_df["fatalities"] > 5) & (observations_df["fatalities"] <= 10)] = 1
    observations_df.fatalities[observations_df["fatalities"] > 10] = 2

    observations_df = observations_df[(observations_df["sex"] == 1) | (observations_df["sex"] == 2)]

    observations_df.to_csv(processed_data_file, index=False)

    return processed_data_file


if __name__ == '__main__':
    preprocessed_file = "../data/preprocessed_data.csv"
    processed_file = "../data/processed_data.csv"
    observation_stats_file = "../data/observation_stats.pickle"
    processed_file = bin_preprocessed_data(preprocessed_file)
    # get_probabilities_for_observations(processed_file, observation_stats_file)
    # initialize_observations_stats()
