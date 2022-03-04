import numpy as np
import torch
import time
torch.set_num_threads(1)

def RungeKutta_LV(x_init, alpha, beta, delta, gamma, yobs, obs_per_day=12):
    nsteps = 1
    tensor_ls = [[0 for i in range(4)] for j in range(nsteps * 240)]
    tensor_ls[0][0] = x_init
    w_ls = [0 for j in range(nsteps * 240)]
    w_ls[0] = x_init
    step_size = 1 / obs_per_day
    for i in range(1, nsteps * 240):
        idx = i - 1
        # k_1
        tensor_ls[i][0] = torch.stack((step_size * (alpha[idx] * w_ls[i-1][0] - beta * w_ls[i-1][0] * w_ls[i-1][1]),
                      step_size * (delta * w_ls[i-1][0] * w_ls[i-1][1] - gamma[idx] * w_ls[i-1][1])))

        w_2 = w_ls[i-1] + tensor_ls[i][0] / 2
        tensor_ls[i][1] = torch.stack((step_size * (alpha[idx] * w_2[0] - beta * w_2[0] * w_2[1]),
                      step_size * (delta * w_2[0] * w_2[1] - gamma[idx] * w_2[1])))

        w_3 = w_ls[i-1] + tensor_ls[i][1] / 2
        tensor_ls[i][2] = torch.stack((step_size * (alpha[idx] * w_3[0] - beta * w_3[0] * w_3[1]),
                      step_size * (delta * w_3[0] * w_3[1] - gamma[idx] * w_3[1])))

        w_4 = w_ls[i-1] + tensor_ls[i][2]
        tensor_ls[i][3] = torch.stack((step_size * (alpha[idx] * w_4[0] - beta * w_4[0] * w_4[1]),
                      step_size * (delta * w_4[0] * w_4[1] - gamma[idx] * w_4[1])))

        w_ls[i] = w_ls[i-1] + (tensor_ls[i][0] + 2 * tensor_ls[i][1] + 2 * tensor_ls[i][2] + tensor_ls[i][3]) / 6

    mse = torch.zeros([240,1])
    for i in range(240):
        mse[i] = torch.sum(torch.square(w_ls[int(nsteps * i)] - yobs[i]))
    return torch.sum(mse)


def RK_solver(use_data_idx, discretization=1, obs_per_day=12):
    start_time = time.time()
    yobs = observations[use_data_idx]
    yobs = torch.tensor(yobs)
    x_init = torch.tensor(yobs[0], requires_grad = True, dtype = torch.float64)
    alpha = torch.tensor(0.6 * np.ones(240), requires_grad=True, dtype = torch.float64)
    gamma = torch.tensor(1. * np.ones(240), requires_grad=True, dtype = torch.float64)
    beta = torch.tensor(0.75, requires_grad=True, dtype = torch.float64)
    delta = torch.tensor(1., requires_grad=True, dtype = torch.float64)
    optimizer = torch.optim.Adam([x_init, alpha, beta, delta, gamma], lr=1e-4) 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    cur_loss = 1e12
    for epoch in range(1000000):
        optimizer.zero_grad()
        # compute loss function
        loss = RungeKutta_LV(x_init, alpha, beta, delta, gamma, yobs)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if epoch % 100 == 0:
            print(epoch, loss.item())
            if torch.isnan(loss) == False and loss.item() - cur_loss > -0.01 and loss.item() - cur_loss < 0.01:
                print(cur_loss, loss.item())
                break
            cur_loss = loss.item()
            
    
    np.save('LV-RK-results/time-' + str(use_data_idx) + '.npy', np.array(time.time() - start_time))
    np.save('LV-RK-results/alpha-' + str(use_data_idx) + '.npy', alpha.detach().numpy())
    np.save('LV-RK-results/xinit-' + str(use_data_idx) + '.npy', x_init.detach().numpy())
    np.save('LV-RK-results/gamma-' + str(use_data_idx) + '.npy', gamma.detach().numpy())
    np.save('LV-RK-results/beta-' + str(use_data_idx) + '.npy', np.array(beta.item()))
    np.save('LV-RK-results/delta-' + str(use_data_idx) + '.npy', np.array(delta.item()))
    return alpha, gamma
    



from multiprocessing import Pool

if __name__ ==  '__main__': 
    torch.set_num_threads(1)
    observations = np.load('LV observations.npy')
    pool = Pool(processes=50)
    results = pool.map(RK_solver, range(100))
    pool.close()
    pool.join()
