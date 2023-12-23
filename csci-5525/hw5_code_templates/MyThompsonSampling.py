import numpy as np

class MyThompsonSampling():
    def __init__(self, num_arms):
        self.num_arms = num_arms

        self.pulled_successes = [0 for i in range(num_arms)]
        self.pulled_fails = [0 for i in range(num_arms)]
        self.max_reward = 0

    def pull_arm(self):
        arm_values = []

        for arm in range(self.num_arms):
            arm_success = self.pulled_successes[arm]
            arm_fail = self.pulled_fails[arm]

            arm_values.append(np.random.beta(arm_success + 1, arm_fail + 1))

        self.selected_arm = np.argmax(arm_values)

        return self.selected_arm

    def update_model(self, reward):

        # we can use == 1 and == 0 for both bernoulli and gaussian since P(1) ~= 0, P(0) ~= 0 in gaussian distribution.
        if reward == 1:
            self.pulled_successes[self.selected_arm] += 1

        elif reward == 0:
            self.pulled_fails[self.selected_arm] += 1

        else:
            if reward > self.max_reward:
                self.max_reward = reward

            reward = reward / self.max_reward

            if reward > 0.5:
                self.pulled_successes[self.selected_arm] += 1

            else:
                self.pulled_fails[self.selected_arm] += 1
