import numpy as np

actions = [0, 1]
epsilon = 0.01

crash_time_steps = 20

episode_end_time_step = 20000

alarm_values = [0, 1]
alarm_probability = 0.7

default_crash_alarm_values = (0, 0)

log_crash_prior = np.log(0.5)

sample_state_keys = ["age", "sex", "alcohol", "drugs", "distracted", "accidents", "fatalities"]

initial_state = 0
states = ["normal", "about_to_crash", "crashed"]

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

project_root = "/Users/BatComp/Desktop/UMass/Courses/687_RL/Project/Car_alarm_trust"
observations_file = project_root + "/data/processed_data.csv"

epsilon_decay_episode = 20
