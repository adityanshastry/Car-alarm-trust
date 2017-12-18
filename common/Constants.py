import numpy as np

actions = [0, 1]
num_features = 9
epsilon = 0.01

crash_time_steps = 20

episode_end_time_step = 20000

alarm_values = [0, 1]
alarm_distribution = [0.9, 0.1]

default_crash_alarm_values = [0, 0]

log_crash_prior = np.log(0.5)
state_keys = ["age", "sex", "alcohol", "drugs", "distracted", "accidents", "fatalities"]

reward_function = {
    0: {
        0: {
            0: 0,
            1: -1
        },
        1: {
            0: -1,
            1: -2
        }
    },
    1: {
        0: {
            0: 0,
            1: -1
        },
        1: {
            0: 1,
            1: 2
        }
    }
}

observations_file = "../data/processed_data.csv"
