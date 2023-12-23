################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
from matplotlib import pyplot as plt

from Environment import Environment

from MyEpsilonGreedy import MyEpsilonGreedy
from MyUCB import MyUCB
from MyThompsonSampling import MyThompsonSampling

num_arms = 8 # Number of arms for each bandit
num_rounds = 500 # Variable 'T' in the writeup
num_repeats = 10 # Variable 'repetitions' in the writeup

# Gaussian environment parameters
means = [7.2, 20.8, 30.4, 10.3, 40.7, 50.1, 1.5, 45.3]
variances = [0.01, 0.02, 0.03, 0.02, 0.04, 0.001, 0.0007, 0.06]

if len(means) != len(variances):
    raise ValueError('Number of means and variances must be the same.')
if len(means) != num_arms or len(variances) != num_arms:
    raise ValueError('Number of means and variances must be equal to the number of arms.')

# Bernoulli environment parameters
p = [0.45, 0.13, 0.71, 0.63, 0.11, 0.06, 0.84, 0.43]

if len(p) != num_arms:
    raise ValueError('Number of Bernoulli probabily values p must be equal to the number of arms.')

# Epsilon-greedy parameter
epsilon = 0.1

if epsilon < 0:
    raise ValueError('Epsilon must be >= 0.')

gaussian_env_params = {'means':means, 'variances':variances}
bernoulli_env_params = {'p':p}

# Use these two objects to simulate the Gaussian and Bernoulli environments.
# In particular, you need to call get_reward() and pass in the arm pulled to receive a reward from the environment.
# Use the other functions to compute the regret.
# See Environment.py for more details. 
gaussian_env = Environment(name='Gaussian', env_params=gaussian_env_params)
bernoulli_env = Environment(name='Bernoulli', env_params=bernoulli_env_params)

#####################
# ADD YOUR CODE BELOW
#####################
from matplotlib import pyplot as plt

# b denotes bernoulli
# g denotes gaussian
eGreedy_b_regret = np.zeros((num_rounds, num_repeats))
ucb_b_regret = np.zeros((num_rounds, num_repeats))
tSampling_b_regret = np.zeros((num_rounds, num_repeats))

eGreedy_g_regret = np.zeros((num_rounds, num_repeats))
ucb_g_regret = np.zeros((num_rounds, num_repeats))
tSampling_g_regret = np.zeros((num_rounds, num_repeats))

bernoulli_regrets = [eGreedy_b_regret, ucb_b_regret, tSampling_b_regret]
gaussian_regrets = [eGreedy_g_regret, ucb_g_regret, tSampling_g_regret]

random_b_regret = np.zeros((num_rounds, num_repeats))
random_g_regret = np.zeros((num_rounds, num_repeats))
for i in range(num_repeats):

    eGreedy_b = MyEpsilonGreedy(num_arms, epsilon)
    ucb_b = MyUCB(num_arms)
    tSampling_b = MyThompsonSampling(num_arms)

    eGreedy_g = MyEpsilonGreedy(num_arms, epsilon)
    ucb_g = MyUCB(num_arms)
    tSampling_g = MyThompsonSampling(num_arms)

    bernoulli_models = [eGreedy_b, ucb_b, tSampling_b]
    gaussian_models = [eGreedy_g, ucb_g, tSampling_g]

    for t in range(num_rounds):
        b_opt_reward = bernoulli_env.get_opt_reward()
        g_opt_reward = gaussian_env.get_opt_reward()

        for j, model in enumerate(bernoulli_models):
            model_selected_arm = model.pull_arm()
            model_reward = bernoulli_env.get_reward(model_selected_arm)
            model_mean_reward = bernoulli_env.get_mean_reward(model_selected_arm)

            if t == 0:
                bernoulli_regrets[j][t, i] = b_opt_reward - model_mean_reward

            else:
                bernoulli_regrets[j][t, i] = bernoulli_regrets[j][t - 1, i] + b_opt_reward - model_mean_reward

            model.update_model(model_reward)

        for j, model in enumerate(gaussian_models):
            model_selected_arm = model.pull_arm()
            model_reward = gaussian_env.get_reward(model_selected_arm)
            model_mean_reward = gaussian_env.get_mean_reward(model_selected_arm)

            if t == 0:
                gaussian_regrets[j][t, i] = g_opt_reward - model_mean_reward

            else:
                gaussian_regrets[j][t, i] = gaussian_regrets[j][t - 1, i] + g_opt_reward - model_mean_reward

            model.update_model(model_reward)

        random_b_selected_arm = np.random.randint(num_arms)
        random_b_reward = bernoulli_env.get_reward(random_b_selected_arm)
        random_b_mean_reward = bernoulli_env.get_mean_reward(random_b_selected_arm)
        if t == 0:
            random_b_regret[t, i] = b_opt_reward - random_b_mean_reward

        else:
            random_b_regret[t, i] = random_b_regret[t - 1, i] + b_opt_reward - random_b_mean_reward

        random_g_selected_arm = np.random.randint(num_arms)
        random_g_reward = gaussian_env.get_reward(random_g_selected_arm)
        random_g_mean_reward = gaussian_env.get_mean_reward(random_g_selected_arm)
        if t == 0:
            random_g_regret[t, i] = g_opt_reward - random_g_mean_reward

        else:
            random_g_regret[t, i] = random_g_regret[t - 1, i] + g_opt_reward - random_g_mean_reward

eGreedy_b_std = np.std(eGreedy_b_regret, axis = 1)
ucb_b_std = np.std(ucb_b_regret, axis = 1)
tSampling_b_std = np.std(tSampling_b_regret, axis = 1)
random_b_std = np.std(random_b_regret, axis = 1)

eGreedy_b_regret = np.mean(eGreedy_b_regret, axis = 1)
ucb_b_regret = np.mean(ucb_b_regret, axis = 1)
tSampling_b_regret = np.mean(tSampling_b_regret, axis = 1)
random_b_regret = np.mean(random_b_regret, axis = 1)

eGreedy_g_std = np.std(eGreedy_g_regret, axis = 1)
ucb_g_std = np.std(ucb_g_regret, axis = 1)
tSampling_g_std = np.std(tSampling_g_regret, axis = 1)
random_g_std = np.std(random_g_regret, axis = 1)

eGreedy_g_regret = np.mean(eGreedy_g_regret, axis = 1)
ucb_g_regret = np.mean(ucb_g_regret, axis = 1)
tSampling_g_regret = np.mean(tSampling_g_regret, axis = 1)
random_g_regret = np.mean(random_g_regret, axis = 1)

T_vec = [i for i in range(num_rounds)]

plt.errorbar(T_vec, eGreedy_b_regret, yerr = eGreedy_b_std, linestyle = "None", label = "Epsilon Greedy Bernoulli")
plt.errorbar(T_vec, ucb_b_regret, yerr = ucb_b_std, linestyle = "None", label = "UCB Bernoulli")
plt.errorbar(T_vec, tSampling_b_regret, yerr = tSampling_b_std, linestyle = "None", label = "Thompson Sampling Bernoulli")
plt.errorbar(T_vec, random_b_regret, yerr = random_b_std, linestyle = "None", label = "Random Sampling Bernoulli")
plt.xlabel("Rounds")
plt.ylabel("Average Cumulative Regret")
plt.legend()

plt.savefig("bernoulli.png")

plt.clf()
plt.errorbar(T_vec, eGreedy_g_regret, yerr = eGreedy_g_std, linestyle = "None", label = "Epsilon Greedy Gaussian")
plt.errorbar(T_vec, ucb_g_regret, yerr = ucb_g_std, linestyle = "None", label = "UCB Gaussian")
plt.errorbar(T_vec, tSampling_g_regret, yerr = tSampling_g_std, linestyle = "None", label = "Thompson Sampling Gaussian")
plt.errorbar(T_vec, random_g_regret, yerr = random_g_std, linestyle = "None", label = "Random Sampling Gaussian")
plt.xlabel("Rounds")
plt.ylabel("Average Cumulative Regret")
plt.legend()

plt.savefig("gaussian.png")
