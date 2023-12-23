import numpy as np

class MyEpsilonGreedy:
    def __init__(self, num_arms, epsilon):
        self.num_arms = num_arms
        self.epsilon = epsilon

        self.pulled_counts = [0 for i in range(num_arms)]
        self.arm_rewards = [0 for i in range(num_arms)]

    # if in epsilon, explore (choose random); if in 1 - epsilon, exploit (choose best)
    def pull_arm(self):
        if np.random.random() < self.epsilon:
            self.selected_arm = np.random.randint(self.num_arms)

        else:
            self.selected_arm = np.argmax(self.arm_rewards)

        self.pulled_counts[self.selected_arm] += 1

        return self.selected_arm

    def update_model(self, reward):
        pulled_times = self.pulled_counts[self.selected_arm]
        self.arm_rewards[self.selected_arm] = (self.arm_rewards[self.selected_arm] * (pulled_times - 1) + reward) / pulled_times
