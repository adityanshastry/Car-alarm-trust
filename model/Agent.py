import numpy as np

from common import Utils, Constants


class Agent:
    def __init__(self, epsilon):
        self.actions = np.array(Constants.actions, dtype=int)
        self.feature_weights = np.zeros(shape=(len(Constants.states), self.actions.shape[0]))
        self.epsilon = epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def reset(self, epsilon):
        self.feature_weights = np.zeros(shape=(len(Constants.states), self.actions.shape[0]))
        self.epsilon = epsilon

    def get_action(self, state):
        # print self.feature_weights
        return np.random.choice(a=self.actions, p=Utils.get_action_distribution(
            np.argmax(self.feature_weights[state, :]), len(self.actions),
            self.epsilon))

    def sarsa_update(self, current_state, current_action, reward, next_state, next_action, lr, gamma):
        delta = reward + gamma * self.feature_weights[next_state][next_action] - self.feature_weights[current_state][
            current_action]
        self.feature_weights[current_state][current_action] += lr * delta

        pass

    def sarsa_terminal_update(self, current_state, current_action, reward, lr):
        delta = reward - self.feature_weights[current_state][current_action]
        self.feature_weights[current_action] += lr * delta

    def q_learning_update(self, current_state, current_action, reward, next_state, lr, gamma):
        max_action_value = np.max(self.feature_weights[next_state, :])
        current_action_value = self.feature_weights[current_state, current_action]

        self.feature_weights[current_state][current_action] += lr * (reward + gamma * max_action_value - current_action_value)
        pass

    def get_params(self):
        return self.feature_weights


def main():
    # agent = Agent(epsilon=Constants.epsilon)
    pass


if __name__ == '__main__':
    main()
