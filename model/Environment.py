import numpy as np

from common import Utils, Constants
import state_sampler


class Environment:
    def __init__(self, observations_file, log_crash_prior, alarm_consistency=True):
        self.time_step = Constants.crash_time_steps
        self.observations_file = observations_file
        self.log_crash_prior = log_crash_prior
        self.alarm_consistency = alarm_consistency
        self.current_state, self.time_to_crash = state_sampler.get_state_with_crash_time(self.observations_file,
                                                                                         self.log_crash_prior)
        self.crash_averted = False
        self.done = False
        self.reward = 0
        pass

    def reset(self):
        self.time_step = Constants.crash_time_steps
        self.current_state, self.time_to_crash = state_sampler.get_state_with_crash_time(self.observations_file,
                                                                                         self.log_crash_prior)
        self.time_step = Constants.crash_time_steps
        self.crash_averted = False
        self.reward = 0
        self.done = False
        pass

    def step(self, current_action):
        self.time_step -= 1

        next_state = np.copy(self.current_state)

        # Set crash to 0 or 1, based on time to crash, current action, and crash averted status
        if self.time_step <= self.time_to_crash and not self.crash_averted:
            if self.current_state[-2] == 0:
                next_state[-2] = 1
                if self.alarm_consistency:
                    next_state[-1] = 1
            elif self.current_state[-2] == 1:
                if current_action == 1:
                    next_state[-2] = 0
                    self.crash_averted = True
                    if self.alarm_consistency:
                        next_state[-1] = 0

        # if alarm is not consistent, randomly set the alarm based on a distribution
        if not self.alarm_consistency:
            next_state[-1] = np.random.choice(a=Constants.alarm_values, p=Constants.alarm_distribution)

        if self.time_step == 0:
            self.done = True

        if self.done:
            if next_state[-2] == 0:
                self.reward = 10
            elif next_state[-2] == 1:
                self.reward = -10
        else:
            self.reward = Constants.reward_function[self.current_state[-2]][self.current_state[-1]][current_action]

        self.current_state = np.copy(next_state)

        assert(len(self.current_state) == Constants.num_features)

        return self.current_state, self.reward, self.done


def main():
    env = Environment("../data/processed_data.csv", Constants.log_crash_prior, False)
    while False or not env.done:
        current_action = np.random.choice(a=[0, 1], p=[0.5, 0.5])
        print env.step(current_action)
    pass


if __name__ == '__main__':
    main()
