import numpy as np

from common import Utils, Constants
import state_sampler


class Environment:
    def __init__(self, observations_file, log_crash_prior, alarm_consistency=True):
        self.current_state = Constants.initial_state
        self.time_step = Constants.crash_time_steps
        self.observations_file = observations_file
        self.log_crash_prior = log_crash_prior
        self.alarm_consistency = alarm_consistency
        self.crash, self.alarm, self.time_to_crash = state_sampler.get_state_with_crash_time(self.observations_file,
                                                                                             self.log_crash_prior)
        self.crash_averted = False
        self.done = False
        self.reward = 0
        pass

    def reset(self):
        self.current_state = Constants.initial_state
        self.time_step = Constants.crash_time_steps
        self.crash, self.alarm, self.time_to_crash = state_sampler.get_state_with_crash_time(self.observations_file,
                                                                                             self.log_crash_prior)
        self.crash_averted = False
        self.done = False
        self.reward = 0
        pass

    def get_state_from_crash_and_alarm(self):
        if self.crash == 0 and self.alarm == 0:
            return 0
        else:
            return 1

    def step(self, current_action):
        self.time_step -= 1

        next_crash, next_alarm = self.crash, self.alarm

        # Set crash to 0 or 1, based on time to crash, current action, and crash averted status
        if self.time_step <= self.time_to_crash and not self.crash_averted:
            if self.crash == 0:
                next_crash = 1
                next_alarm = 1
            elif self.crash == 1:
                if current_action == 1:
                    next_crash = 0
                    self.crash_averted = True
                    next_alarm = 0

        # if alarm is not consistent, randomly set the alarm based on a distribution
        alarm_distribution = np.ones(shape=len(Constants.alarm_values)) * (1 - Constants.alarm_probability)
        alarm_distribution[next_alarm] = Constants.alarm_probability
        if not self.alarm_consistency:
            next_alarm = np.random.choice(a=Constants.alarm_values, p=alarm_distribution)

        if self.time_step == 0 or self.crash_averted:
            self.done = True

        # self.reward = Constants.reward_function[self.crash][self.alarm][current_action]
        self.crash, self.alarm = next_crash, next_alarm

        if self.done:
            if next_crash == 0:
                self.current_state = self.get_state_from_crash_and_alarm()
                if self.current_state == 0:
                    self.reward = 1
                elif self.current_state == 1:
                    self.reward = -1
            elif next_crash == 1:
                self.current_state = 2
                self.reward = -1
        else:
            self.current_state = self.get_state_from_crash_and_alarm()

        return self.current_state, self.reward, self.done


def main():
    env = Environment("../data/processed_data.csv", Constants.log_crash_prior)
    print env.time_to_crash
    while False or not env.done:
        # current_action = np.random.choice(a=[0, 1], p=[0.5, 0.5])
        current_action = 1
        print env.step(current_action)
    pass


if __name__ == '__main__':
    main()
