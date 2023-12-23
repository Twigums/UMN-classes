import numpy as np

class MyUCB():
    def __init__(self, num_arms):
        self.num_arms = num_arms

        self.pulled_counts = [0 for i in range(num_arms)]
        self.arm_rewards = [0 for i in range(num_arms)]
        self.total_pulls = 0

    def pull_arm(self):
        # init condition
        if min(self.pulled_counts) == 0:
            self.selected_arm = np.argmin(self.pulled_counts)

        else:
            # variance term
            weighted_arm_rewards = []
            for i in range(len(self.arm_rewards)):
                weighted_arm_rewards.append(self.arm_rewards[i] + np.sqrt((2 * np.log(self.total_pulls)) / self.pulled_counts[i]))

            self.selected_arm = np.argmax(weighted_arm_rewards)

        self.pulled_counts[self.selected_arm] += 1
        self.total_pulls += 1

        return self.selected_arm

    def update_model(self, reward):
        pulled_times = self.pulled_counts[self.selected_arm]
        self.arm_rewards[self.selected_arm] = (self.arm_rewards[self.selected_arm] * (pulled_times - 1) + reward) / pulled_times
