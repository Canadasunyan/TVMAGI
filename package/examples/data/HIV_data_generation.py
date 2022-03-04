import numpy as np
import torch
import scipy

days = 100  
obs_per_day = 1
nobs = days * obs_per_day
linspace = 1000
X = np.arange(0, days, 1./(obs_per_day * linspace))
true_lambda = 36.
true_rho = 0.108 
k = 5e-4
true_gamma = np.cos(3.14 * X / 500)
true_delta = 0.1
N = 1000.
c = 3.5

def gen_data(idx, true_lambda, true_rho, true_gamma, true_delta, days, obs_per_day, state0=[350., 20., 1200.],N=N, k=k, c=c, linspace=1000, noise=True):
    """
    generate data using pre-set parameters 
    """
    # check input
    # [ S, E, I, D, cfr0]
    nobs = days
    step_size = 1. / (linspace * obs_per_day)
    state_ls = np.ones((days * linspace * obs_per_day, 3))
    state_ls[0] = state0.copy()
    for i in range(1, linspace * days * obs_per_day):
        index = i - 1
        state_ls[i][0] = state_ls[i - 1][0] + step_size * (true_lambda - true_rho * state_ls[i - 1][0] - k * (1-true_gamma[index]) * state_ls[i - 1][0] * state_ls[i - 1][2])
        state_ls[i][1] = state_ls[i - 1][1] + step_size * (k * (1-true_gamma[index]) * state_ls[i - 1][0] * state_ls[i - 1][2] - true_delta * state_ls[i - 1][1])
        state_ls[i][2] = state_ls[i - 1][2] + step_size * (N * true_delta * state_ls[i - 1][1] - c * state_ls[i - 1][2]) 
    states = state_ls[::obs_per_day * linspace]
    if noise == True:
        yobs = states.copy()
        np.random.seed(idx)
        yobs[:, 2] = states[:, 2] * np.random.normal(np.repeat(1., nobs), np.linspace(0.03, 0.03, nobs))

    return yobs[:, 0], np.log(yobs[:, 2])


observations = np.zeros((100, 100, 2))
for i in range(100):
    print(i)
    Tt, yobs = gen_data(i, true_lambda, true_rho, true_gamma, true_delta, days, obs_per_day, noise=True)
    observations[i][:, 0] = yobs
    observations[i][:, 1] = Tt
    
np.save('HIV_observations.npy', observations)
