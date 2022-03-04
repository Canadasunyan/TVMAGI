import numpy as np

days = 20  
obs_per_day = 12
nobs = 240
X = np.arange(0, days, 1./(obs_per_day * 1000))
alpha = 0.6 + 0.3 * np.cos(6.28 * X / 10)
beta = 0.75 
delta = 1. 
gamma = 1 + 0.1 * np.sin(6.28 * X / 10)
theta_true = np.vstack([alpha, gamma]).T

# data generation for 20 years with monthly observations
def recover_data(alpha, beta, delta, gamma, days, obs_per_day, state0=[3., 1.], linspace=1000, noise=True):
    nobs = days
    step_size = 1. / (linspace * obs_per_day)
    state_ls = np.ones((days * linspace * obs_per_day, 2))
    state_ls[0] = state0.copy()
    for i in range(1, linspace * days * obs_per_day):
        index = i - 1
        state_ls[i][0] = state_ls[i - 1][0] + step_size * (
                    alpha[index] * state_ls[i - 1][0] - beta * state_ls[i - 1][0] * state_ls[i - 1][1])
        state_ls[i][1] = state_ls[i - 1][1] + step_size * (
                    delta * state_ls[i - 1][0] * state_ls[i - 1][1] - gamma[index]  * state_ls[i - 1][1])
    states = state_ls[::linspace]

    return states

true_x = recover_data(alpha, beta, delta, gamma, days, obs_per_day)

def gen_data(idx, alpha, beta, delta, gamma, days, obs_per_day, state0=[3., 1.], linspace=1000, noise=True):
    nobs = days
    step_size = 1. / (linspace * obs_per_day)
    state_ls = np.ones((days * linspace * obs_per_day, 2))
    state_ls[0] = state0.copy()
    for i in range(1, linspace * days * obs_per_day):
        index = i - 1
        state_ls[i][0] = state_ls[i - 1][0] + step_size * (
                    alpha[index] * state_ls[i - 1][0] - beta * state_ls[i - 1][0] * state_ls[i - 1][1])
        state_ls[i][1] = state_ls[i - 1][1] + step_size * (
                    delta * state_ls[i - 1][0] * state_ls[i - 1][1] - gamma[index]  * state_ls[i - 1][1])
    states = state_ls[::linspace]
    np.random.seed(idx)
    states *= np.random.normal(1, 0.03, states.shape)
    return states

obs = np.zeros((100, 240, 2))
for i in range(100):
    obs[i] = gen_data(i, alpha, beta, delta, gamma, days, obs_per_day)
    
np.save('LV_observations.npy', obs)
