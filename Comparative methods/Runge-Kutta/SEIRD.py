import numpy as np
import torch
import time
torch.set_num_threads(1)
torch.set_default_dtype(torch.double)

def RungeKutta_SEIR(x_init, beta, ve, vi, pd, yobs):
    discretization = 2
    days = 32
    step_size = 1 / discretization
    tensor_ls = [[0 for i in range(4)] for j in range(discretization * days)]
    tensor_ls[0][0] = x_init
    w_ls = [0 for j in range(discretization * days)]
    w_ls[0] = x_init
    for i in range(1, discretization * days):
        idx = int(i)
        # k_1
        tensor_ls[i][0] = torch.stack((-step_size * beta[idx] * torch.exp(w_ls[i-1][2]) / 100000,
                      step_size * beta[idx] * torch.exp(w_ls[i-1][0]+w_ls[i-1][2]-w_ls[i-1][1]) / 100000 -
                                    step_size * ve[idx],
                      step_size * ve[idx] * torch.exp(w_ls[i-1][1] - w_ls[i-1][2]) -
                                    step_size * vi,
                      step_size * pd[idx] * vi * torch.exp(w_ls[i-1][2] - w_ls[i-1][3])))
        # k_2
        w2 = w_ls[i-1] + tensor_ls[i][0] / 2
        tensor_ls[i][1] = torch.stack((-step_size * beta[idx] * torch.exp(w2[2]) / 100000,
                      step_size * beta[idx] * torch.exp(w2[0]+w2[2]-w2[1]) / 100000 -
                                    step_size * ve[idx],
                      step_size * ve[idx] * torch.exp(w2[1] - w2[2]) -
                                    step_size * vi,
                      step_size * pd[idx] * vi * torch.exp(w2[2] - w2[3])))
        # k_3
        w3 = w_ls[i-1] + tensor_ls[i][1] / 2
        tensor_ls[i][2] = torch.stack((-step_size * beta[idx] * torch.exp(w3[2]) / 100000,
                      step_size * beta[idx] * torch.exp(w3[0]+w3[2]-w3[1]) / 100000 -
                                    step_size * ve[idx],
                      step_size * ve[idx] * torch.exp(w3[1] - w3[2]) -
                                    step_size * vi,
                      step_size * pd[idx] * vi * torch.exp(w3[2] - w3[3])))
        # k_3
        w4 = w_ls[i-1] + tensor_ls[i][2]
        tensor_ls[i][3] = torch.stack((-step_size * beta[idx] * torch.exp(w4[2]) / 100000,
                      step_size * beta[idx] * torch.exp(w4[0]+w4[2]-w4[1]) / 100000 -
                                    step_size * ve[idx],
                      step_size * ve[idx] * torch.exp(w4[1] - w4[2]) -
                                    step_size * vi,
                      step_size * pd[idx] * vi * torch.exp(w4[2] - w4[3])))
        w_ls[i] = w_ls[i-1] + (tensor_ls[i][0] + 2 * tensor_ls[i][1] + 2 * tensor_ls[i][2] + tensor_ls[i][3]) / 6
    mse = torch.zeros([days * discretization,1])
    for i in range(days):
        mse[i] = torch.sum(torch.square(w_ls[int(discretization * i)] - yobs[i]))
    return torch.sum(mse)


def RK_solver(use_data_idx, days=32, discretization=2):
    start_time = time.time()
    yobs = observations[use_data_idx]
    yobs[:, 1] = np.interp(np.arange(0, days, 1), np.arange(0, days, 2), yobs[::2, 1])
    yobs = torch.tensor(yobs)
    x_init = torch.tensor(yobs[0], requires_grad = True, dtype = torch.float64)
    beta = torch.tensor(1.8 * np.ones(days * discretization), requires_grad=True, dtype = torch.float64)
    ve = torch.tensor(0.1 * np.ones(days * discretization), requires_grad=True, dtype = torch.float64)
    pd = torch.tensor(0.2 * np.ones(days * discretization), requires_grad=True, dtype = torch.float64)
    vi = torch.tensor(0.1, requires_grad=True, dtype = torch.float64)
    optimizer = torch.optim.Adam([x_init, beta, ve, vi, pd], lr=1e-4)  # , weight_decay = 1.0
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    cur_loss = float('inf')
    for epoch in range(10000):
        optimizer.zero_grad()
        # compute loss function
        loss = RungeKutta_SEIR(x_init, beta, ve, vi, pd, torch.tensor(yobs))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if epoch % 100 == 0:
            print(epoch, loss.item())
            if torch.isnan(loss) == False and loss.item() - cur_loss > -0.001 and loss.item() - cur_loss < 0.001:
                print(cur_loss, loss.item())
                break
            cur_loss = loss.item()
            
    
    np.save('beta-' + str(use_data_idx) + '.npy', beta.detach().numpy())
    np.save('xinit-' + str(use_data_idx) + '.npy', x_init.detach().numpy())
    np.save('ve-' + str(use_data_idx) + '.npy', ve.detach().numpy())
    np.save('pd-' + str(use_data_idx) + '.npy', pd.detach().numpy())
    np.save('vi-' + str(use_data_idx) + '.npy', np.array(vi.item()))
    
    



from multiprocessing import Pool

if __name__ ==  '__main__': 
    torch.set_num_threads(1)
    observations = np.load('SEIRD_observations.npy')
    N = 100000.
    pool = Pool(processes=100)
    print('OK')
    results = pool.map(RK_solver, range(100))
    pool.close()
    pool.join()
