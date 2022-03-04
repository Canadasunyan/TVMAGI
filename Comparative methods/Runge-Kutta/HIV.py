import numpy as np
import torch
import time
from multiprocessing import Pool
torch.set_num_threads(1)

def RungeKutta_HIV(x_init, a1, a2, yobs, obs_per_day=1, days=100):
    global Tt
    Tt_torch = torch.tensor(Tt)
    c = 3.5
    nsteps = 100
    tensor_ls = [[0 for i in range(4)] for j in range(nsteps * days)]
    w_ls = [0 for j in range(nsteps * days)]
    w_ls[0] = x_init
    step_size = 1 / nsteps
    for i in range(1, nsteps * days):
        idx = int(i / nsteps)
        # k_1
        tensor_ls[i][0] = step_size * (1e5 * a1[idx] * torch.exp(-w_ls[i-1]) + 1e3 * a2[idx] * Tt_torch[idx] * torch.exp(-w_ls[i-1]) - c)

        w_2 = w_ls[i-1] + tensor_ls[i][0] / 2
        tensor_ls[i][1] = step_size * (1e5 * a1[idx] * torch.exp(-w_2) + 1e3 * a2[idx] * Tt_torch[idx] * torch.exp(-w_2) - c)

        w_3 = w_ls[i-1] + tensor_ls[i][1] / 2
        tensor_ls[i][2] = step_size * (1e5 * a1[idx] * torch.exp(-w_3) + 1e3 * a2[idx] * Tt_torch[idx] * torch.exp(-w_3) - c)

        w_4 = w_ls[i-1] + tensor_ls[i][2]
        tensor_ls[i][3] = step_size * (1e5 * a1[idx] * torch.exp(-w_4) + 1e3 * a2[idx] * Tt_torch[idx] * torch.exp(-w_4) - c)

        w_ls[i] = w_ls[i-1] + (tensor_ls[i][0] + 2 * tensor_ls[i][1] + 2 * tensor_ls[i][2] + tensor_ls[i][3]) / 6
    mse = torch.zeros([days,1])
    for i in range(days):
        mse[i] = torch.sum(torch.square(w_ls[int(nsteps * i)] - yobs[i]))
    print(torch.sum(mse))
    return torch.sum(mse)


def RK_solver(use_data_idx, discretization=1, obs_per_day=1, days=100):
    global observations
    start_time = time.time()
    yobs = observations[use_data_idx]
    x_init = torch.tensor(yobs[0], requires_grad = True, dtype = torch.float64)
    yobs = torch.tensor(yobs)
    a1 = torch.tensor(0.18 * np.ones(days), requires_grad=True, dtype = torch.float64)
    a2 = torch.tensor(-0.05 * np.ones(days), requires_grad=True, dtype = torch.float64)
    optimizer = torch.optim.Adam([x_init, a1, a2], lr=1e-4) 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    cur_loss = 1e12
    for epoch in range(1000000):
        optimizer.zero_grad()
        # compute loss function
        loss = RungeKutta_HIV(x_init, a1, a2, yobs)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if epoch % 100 == 0:
            print(epoch, loss.item())
            if torch.isnan(loss) == False and loss.item() - cur_loss > -0.01 and loss.item() - cur_loss < 0.01:
                print(cur_loss, loss.item())
                break
            cur_loss = loss.item()
            
    
    np.save('HIV-RK-results/time-' + str(use_data_idx) + '.npy', np.array(time.time() - start_time))
    np.save('HIV-RK-results/a1-' + str(use_data_idx) + '.npy', a1.detach().numpy())
    np.save('HIV-RK-results/xinit-' + str(use_data_idx) + '.npy', x_init.detach().numpy())
    np.save('HIV-RK-results/a2-' + str(use_data_idx) + '.npy', a2.detach().numpy())
    





if __name__ ==  '__main__': 
#     torch.set_num_threads(1)
#     torch.set_num_threads(1)
    days = 100
    data = np.load('HIV observations.npy')
    Tt = data[0][:, 1]
    observations = np.zeros((100, 100, 1))
    for i in range(100):
        observations[i] = data[i, :, 0].reshape(-1, 1)
        
    pool = Pool(processes=50)
    results = pool.map(RK_solver, range(100))
    pool.close()
    pool.join()
