import numpy as np

from common import Utils, Constants


class Agent:
    def __init__(self, epsilon):
        self.actions = np.array(Constants.actions, dtype=int)
        self.feature_weights = np.zeros(shape=(len(self.actions), Constants.num_features))
        self.epsilon = epsilon

    def reset(self):
        self.feature_weights = np.zeros(shape=(len(self.actions), Constants.num_features))

    def get_action(self, current_state):
        return np.random.choice(a=self.actions, p=Utils.get_action_distribution(
            np.argmax(np.dot(self.feature_weights, current_state)), len(self.actions),
            self.epsilon))

    def get_action_value(self, current_state, current_action):
        return np.dot(self.feature_weights[current_action], current_state)

    def sarsa_update(self, current_state, current_action, reward, next_state, next_action, lr, gamma):
        delta = reward + gamma * self.get_action_value(next_state, next_action) - self.get_action_value(current_state,
                                                                                                        current_action)
        self.feature_weights[current_action] += lr * delta * current_state

        pass

    def q_learning_update(self, current_state, current_action, reward, next_state, lr, gamma):
        max_action_value = np.max(np.dot(self.feature_weights, next_state))
        current_action_value = self.get_action_value(current_state, current_action)

        self.feature_weights[current_action] += lr * (reward + gamma * max_action_value - current_action_value) * \
                                                current_state

        pass


def main():
    # agent = Agent(epsilon=Constants.epsilon)
    pass


if __name__ == '__main__':
    main()
