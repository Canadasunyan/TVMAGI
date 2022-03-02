import numpy as np
import torch
import scipy
import argparse
import sys
from arma import ode_system, solve_magi
from arma import matrix
import time
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize # import figsize
#figsize(12.5, 4) # 设置 figsize
plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率
torch.set_num_threads(1)





# Derivatves of X according to the ODE structure
def fOde(theta, x):
    """
    theta: list[4]: beta, ve, vi, pd
    x: array(n, 4)
    r: array(n, 2)
    """
    global N
    logS = x[:, 0]
    logE = x[:, 1]
    logI = x[:, 2]
    logD = x[:, 3]
    logSdt = -theta[0] * np.exp(logI) / N  # (1)
    logEdt = theta[0] * np.exp(logS + logI - logE) / N - theta[1]  # (2)
    logIdt = np.exp(logE - logI) * theta[1] - theta[2]  # (3)
    logDdt = np.exp(logI - logD) * 0.25 * theta[3] * theta[2]  # (4)
    return np.stack([logSdt, logEdt, logIdt, logDdt], axis=1)


# Derivatives of X
def fOdeDx(theta, x):
    """
    returns derivation of x given theta
    theta: list[4]
    x: array(n, 4)
    r: array(n, 4, 4)
    """
    resultDx = np.zeros(shape=[np.shape(x)[0], np.shape(x)[1], np.shape(x)[1]])
    global N
    logS = x[:, 0]
    logE = x[:, 1]
    logI = x[:, 2]
    logD = x[:, 3]
    # [:, i, j]: 第j个方程关于第i个状态求导
    # (1) / dI
    resultDx[:, 2, 0] = -theta[0] * np.exp(logI) / N
    # (1) / dS, (1) /dE, (1) / dD = 0
    # (2) / dS
    resultDx[:, 0, 1] = theta[0] * np.exp(logS + logI - logE) / N
    # (2) / dE
    resultDx[:, 1, 1] = -theta[0] * np.exp(logS + logI - logE) / N
    # (2) / dI
    resultDx[:, 2, 1] = theta[0] * np.exp(logS + logI - logE) / N
    # (2) / dD = 0
    # (3) / dS = 0
    # (3) / dE
    resultDx[:, 1, 2] = np.exp(logE - logI) * theta[1]
    # (3) / dI
    resultDx[:, 2, 2] = -np.exp(logE - logI) * theta[1]
    # (3) / dD = 0, (4) / dS, dE = 0
    # (4) / dI
    resultDx[:, 2, 3] = np.exp(logI - logD) * 0.25 * theta[3] * theta[2]
    # (4) / dD
    resultDx[:, 3, 3] = -np.exp(logI - logD) * 0.25 * theta[3] * theta[2]
    return resultDx


def fOdeDtheta(theta, x):
    """
    returns derivation of theta given x
    theta: list[4]
    x: array(n, 4)
    r: array(n, 4, 4)
    """
    global N
    resultDtheta = np.zeros(shape=[np.shape(x)[0], np.shape(theta)[0], np.shape(x)[1]])
    logS = x[:, 0]
    logE = x[:, 1]
    logI = x[:, 2]
    logD = x[:, 3]
    # [:, i, j]: 第j个方程对theta_i求导
    # (1) / dRe
    resultDtheta[:, 0, 0] = -np.exp(logI) / N
    # (2) / d theta[0]
    resultDtheta[:, 0, 1] = np.exp(logS + logI - logE) / N
    # (2) / theta[1]
    resultDtheta[:, 1, 1] = -1.
    # (3) / dtheta[1]
    resultDtheta[:, 1, 2] = np.exp(logE - logI)
    # (3) / dtheta[2]
    resultDtheta[:, 2, 2] = -1.
    # (4) / theta[2]
    resultDtheta[:, 2, 3] = np.exp(logI - logD) * 0.25 * theta[3] 
    # (4) / theta[3]
    resultDtheta[:, 3, 3] = np.exp(logI - logD) * 0.25 * theta[2]
    return resultDtheta

def fOdeTorch(theta, x, constant_param_ls):
    """
    theta: list[4]: beta, ve, vi, pd
    x: array(n, 4)
    r: array(n, 2)
    """
    global N
    logS = x[:, 0]
    logE = x[:, 1]
    logI = x[:, 2]
    logD = x[:, 3]
    logSdt = -theta[:, 0] * torch.exp(logI) / N  # (1)
    logEdt = theta[:, 0] * torch.exp(logS + logI - logE) / N - theta[:, 1]  # (2)
    logIdt = torch.exp(logE - logI) * theta[:, 1] - constant_param_ls[0]  # (3)
    # reparametrize on pd
    logDdt = torch.exp(logI - logD) * 0.25 * theta[:, 2] * constant_param_ls[0]  # (4)
    return torch.stack([logSdt, logEdt, logIdt, logDdt], axis=1)


def copy_mat(arma_mat):
    return np.copy(matrix(arma_mat).reshape([-1])).reshape([arma_mat.n_rows, arma_mat.n_cols])

def pointwisethetasigmallikTorch(xlatent, theta, time_constant_param_ls, sigma, inferred_theta, ydata, CovAllDimensionsPyList, fOdeTorch,
                                 priorTemperature, obs_per_day, positive_param=True):
    # length of observed y (t)
    n = ydata.shape[0]
    pdimension = ydata.shape[1]
    thetadimension = theta.shape[1]
    sigmaSq = torch.pow(sigma, 2)
    fderiv = fOdeTorch(theta, xlatent, time_constant_param_ls)
    res = torch.zeros([pdimension, 3]).double()
    fitDerivError = torch.zeros([n, pdimension]).double()
    nobs = torch.zeros([pdimension]).double()
    fitLevelErrorSumSq = torch.zeros([pdimension]).double()
    for vEachDim in range(pdimension):
        fitDerivError[:, vEachDim] = fderiv[:, vEachDim]
        fitDerivError[:, vEachDim] -= CovAllDimensionsPyList[vEachDim]['mphi'] @ xlatent[:, vEachDim]
        nobs[vEachDim] = torch.sum(torch.isfinite(ydata[:, vEachDim]))
        obsIdx = torch.isfinite(ydata[:, vEachDim])
        fitLevelErrorSumSq[vEachDim] = torch.sum(torch.square(xlatent[obsIdx, vEachDim] - ydata[obsIdx, vEachDim]))
    res[:, 0] = -0.5 * fitLevelErrorSumSq / sigmaSq - torch.log(sigma + 0.001) * nobs
    res[:, 0] /= priorTemperature[2]
    KinvfitDerivError = torch.zeros([n, pdimension]).double()
    CinvX = torch.zeros([n, pdimension]).double()
    for vEachDim in range(pdimension):
        # inverse of K
        KinvfitDerivError[:, vEachDim] = CovAllDimensionsPyList[vEachDim]['Kinv'] @ fitDerivError[:, vEachDim]
        # inverse of Cd
        CinvX[:, vEachDim] = CovAllDimensionsPyList[vEachDim]['Cinv'] @ xlatent[:, vEachDim]
    res[:, 1] = -0.5 * torch.sum(fitDerivError * KinvfitDerivError, dim=0) / priorTemperature[0]
    #  prior distriobution of X
    res[:, 2] = -0.5 * torch.sum(xlatent * CinvX, dim=0) / priorTemperature[1]
    theta_lb = torch.clamp(theta[:, 2], min=0.)
#     theta_ub = torch.clamp(theta[:, 2], max=0.3)
    return torch.sum(res) - 1e6 * torch.sum(torch.square(theta[:, 2] - theta_lb))

def MaternKernel(d, phi_1, phi_2, nu=2.5):
    """
    construct a kernel given time points and hyper parameters
    """
    if nu == 2.5:  
        a = torch.square(phi_1) * (
                1. + np.sqrt(5) * d / phi_2 + 5. * torch.square(d) / (3. * torch.square(phi_2))) * torch.exp(
            -np.sqrt(5) * d / phi_2)
        return a.double()
    else:
        a = torch.square(phi_1) * (1. + np.sqrt(3) * d / phi_2) * torch.exp(-np.sqrt(3) * d / phi_2)       
        return a.double()


def kernelllik(phi_1, phi_2, sigma, y, d_matrix, phi1_lb, phi2_lb, sigma_lb):
    """
    optimize the kernel hyperparameters by maximizing marginal likelihood
    """
    phi_1_bounded = torch.clamp(phi_1, min=phi1_lb)
    phi_2_bounded = torch.clamp(phi_2, min=phi2_lb)
    sigma_bounded = torch.clamp(sigma, min=sigma_lb)
    K = MaternKernel(d_matrix, phi_1, phi_2)
    K += torch.square(sigma) * torch.eye(y.shape[0]).double()
    return -y.shape[0] * np.log(np.sqrt(2 * np.pi)) - 0.5 * y @ torch.inverse(K) @ y - 0.5 * torch.logdet(K) - 1e8 * torch.square(phi_1 - phi_1_bounded) - 1e8 * torch.square(phi_2 - phi_2_bounded) - 1e8 * torch.square(sigma - sigma_bounded)

def xthetasigmallikTorch(xlatent, theta, time_constant_param_ls, sigma, inferred_theta, ydata, CovAllDimensionsPyList, fOdeTorch,
                         priorTemperature,
                         KinvthetaList, positive=True):
    # length of observed y (t)
    n = ydata.shape[0]
    pdimension = ydata.shape[1]
    thetadimension = theta.shape[1]
    sigmaSq = torch.pow(sigma, 2)
    fderiv = fOdeTorch(theta, xlatent, time_constant_param_ls)
    res = torch.zeros([pdimension, 3]).double()
    res_theta = torch.zeros(thetadimension).double()
    res2 = torch.zeros(1).double()
    fitDerivError = torch.zeros([n, pdimension]).double()
    nobs = torch.zeros([pdimension]).double()
    fitLevelErrorSumSq = torch.zeros([pdimension]).double()
    for vEachDim in range(pdimension):
        fitDerivError[:, vEachDim] = fderiv[:, vEachDim]
        fitDerivError[:, vEachDim] -= CovAllDimensionsPyList[vEachDim]['mphi'] @ xlatent[:, vEachDim]
        nobs[vEachDim] = torch.sum(torch.isfinite(ydata[:, vEachDim]))
        obsIdx = torch.isfinite(ydata[:, vEachDim])
        fitLevelErrorSumSq[vEachDim] = torch.sum(torch.square(xlatent[obsIdx, vEachDim] - ydata[obsIdx, vEachDim]))
    res[:, 0] = -0.5 * fitLevelErrorSumSq / sigmaSq - torch.log(sigma + 0.0001) * nobs
    res[:, 0] /= priorTemperature[2]
    KinvfitDerivError = torch.zeros([n, pdimension]).double()
    CinvX = torch.zeros([n, pdimension]).double()
    for vEachDim in range(pdimension):
        # inverse of K
        KinvfitDerivError[:, vEachDim] = CovAllDimensionsPyList[vEachDim]['Kinv'] @ fitDerivError[:, vEachDim]
        # inverse of Cd
        CinvX[:, vEachDim] = CovAllDimensionsPyList[vEachDim]['Cinv'] @ xlatent[:, vEachDim]
    for thetaEachDim in range(thetadimension):
        res_theta[thetaEachDim] = -0.5 * torch.sum(
            (theta[:, thetaEachDim] - inferred_theta[thetaEachDim]) @ KinvthetaList[thetaEachDim] @ (
                    theta[:, thetaEachDim] - inferred_theta[thetaEachDim]))
    res[:, 1] = -0.5 * torch.sum(fitDerivError * KinvfitDerivError, dim=0) / priorTemperature[0]
    res[:, 2] = -0.5 * torch.sum(xlatent * CinvX, dim=0) / priorTemperature[1]
    theta_lb = torch.clamp(theta[:, 2], min = 0.)
    return torch.sum(res) + torch.sum(res_theta)  - 1e6 * torch.sum(torch.square(theta[:, 2] - theta_lb))



def TVMAGI_solver(use_data_idx, fOde=fOde, fOdeDx=fOdeDx, fOdeDtheta=fOdeDtheta,
                  days = 32, 
                  discretization = 4,
                  obs_per_day = 1, 
                  theta_lowerbound=np.array([0., 0., 0., 0.]),
                  theta_upperbound=np.array([np.inf, 1., 1., 1.]),
                  param_names = ['re', 've', 'vi', 'pd'],
                  is_time_varying=[True, True, False, True], 
                  use_trajectory='inferred', 
                  learning_rate=np.array([1e-4, 1e-3, 1e-5]), 
                  n_iter = [15001, 80000, 15000, 60000],
                  phi1_lb_ls=np.array([2., 0.1, 0.1]), 
                  phi2_lb_ls=np.array([8., 8., 8.]), 
                  sigma_lb_ls = np.array([0.01, 0.01, 0.01])):
    
    yobs = observations[use_data_idx]
    yobs[:, 1] = np.interp(np.arange(0, days, 1), np.arange(0, days, 2), yobs[::2, 1])
    start_time = time.time()
    nobs, p_dim = yobs.shape[0], yobs.shape[1]
    n_points = days * discretization
    theta_dim = theta_lowerbound.shape[0]
    d_matrix = torch.zeros((n_points, n_points), dtype=torch.double)
    for i in range(n_points):
        for j in range(n_points):
            if i > j:
                d_matrix[i][j] = (i - j) / (obs_per_day * discretization)
            else:
                d_matrix[i][j] = (j - i) / (obs_per_day * discretization)
    Ode_system = ode_system("ODE-python", fOde, fOdeDx, fOdeDtheta,
                             thetaLowerBound=theta_lowerbound,
                             thetaUpperBound=theta_upperbound)
    tvecObs = np.arange(0, days, 1. / obs_per_day)
    tvecFull = np.arange(0, days, 1. / (obs_per_day * discretization))
    yFull = np.ndarray([n_points, p_dim])
    yFull.fill(np.nan)
    yFull[np.arange(0, discretization * nobs, discretization).astype(int), :] = yobs
    xInitExogenous = np.zeros_like(yFull)
    # interpolate: find the ydata of tvecFull given observations
    for i in range(p_dim):
        xInitExogenous[:, i] = np.interp(tvecFull, tvecObs, yobs[:, i])
        
        
    # First stage: use MAGI package to optimize theta as constant #####################
    
    
    result = solve_magi(
        yFull,
        Ode_system,
        tvecFull,
        sigmaExogenous=np.array([]),
        phiExogenous=np.array([[]]),
        xInitExogenous=xInitExogenous,
        thetaInitExogenous=np.array([]),
        muExogenous=np.array([[]]),
        dotmuExogenous=np.array([[]]),
        priorTemperatureLevel=yFull.shape[0] / yobs.shape[0],
        priorTemperatureDeriv=yFull.shape[0] / yobs.shape[0],
        priorTemperatureObs=1.0,
        kernel="generalMatern",
        nstepsHmc=100,
        burninRatioHmc=0.5,
        niterHmc=n_iter[0],
        stepSizeFactorHmc=0.01,
        nEpoch=1,
        bandSize=40,
        useFrequencyBasedPrior=True,
        useBand=False,
        useMean=False,
        useScalerSigma=False,
        useFixedSigma=False,
        verbose=True)
    samplesCpp = result['samplesCpp']
    llikId = 0
    xId = range(np.max(llikId) + 1, np.max(llikId) + yFull.size + 1)
    # dimension of theta
    thetaId = range(np.max(xId) + 1, np.max(xId) + theta_dim + 1)
    sigmaId = range(np.max(thetaId) + 1, np.max(thetaId) + yFull.shape[1] + 1)
    burnin = int(n_iter[0] * 0.5)
    xsampled = samplesCpp[xId, (burnin + 1):]
    xsampled = xsampled.reshape([yFull.shape[1], yFull.shape[0], -1])
    CovAllDimensionsPyList = []
    thetaSampled = samplesCpp[thetaId, (burnin + 1):]
    inferred_theta = np.mean(thetaSampled, axis=-1)
    print(inferred_theta)
    sigmaSampled = samplesCpp[sigmaId, (burnin + 1):]
    inferred_sigma = np.mean(sigmaSampled, axis=-1)
    inferred_trajectory = np.mean(xsampled, axis=-1)
    for each_gpcov in result['result_solved'].covAllDimensions:
            each_pycov = dict(
                Cinv=torch.from_numpy(matrix(each_gpcov.Cinv)).double(),
                Kinv=torch.from_numpy(matrix(each_gpcov.Kinv)).double(),
                mphi=torch.from_numpy(matrix(each_gpcov.mphi)).double(),
            )
            CovAllDimensionsPyList.append(each_pycov)


    # Pointwise optimization ###################################
    
    
    TV_theta_mean = np.zeros(int(sum(is_time_varying)))
    tv_index = 0
    for thetaEachDim in range(theta_dim):
        if is_time_varying[thetaEachDim] == True:
            TV_theta_mean[tv_index] = inferred_theta[thetaEachDim]
            tv_index += 1

    if use_trajectory == 'observation':
        pointwise_xlatent_torch = torch.tensor(xInitExogenous, requires_grad=True, dtype=torch.double)
    elif use_trajectory == 'inferred':
        pointwise_xlatent_torch = torch.tensor(inferred_trajectory.transpose(), requires_grad=True, dtype=torch.double)
    else:
        raise ValueError
    tmp1 = np.array([TV_theta_mean])
    initial_tvtheta = np.repeat(tmp1, pointwise_xlatent_torch.shape[0], axis=0)
    pointwise_theta_torch = torch.tensor(initial_tvtheta, requires_grad=True, dtype=torch.double)
    sigma_torch = torch.tensor(inferred_sigma, requires_grad=True, dtype=torch.double)
    time_constant_param_ls = []
    for thetaEachDim in range(theta_dim):
        if is_time_varying[thetaEachDim] == 0:
            param_name = param_names[thetaEachDim]
            locals()[param_name] = torch.tensor([inferred_theta[thetaEachDim]], requires_grad=True, dtype=torch.double)
            time_constant_param_ls.append(eval(param_name))

    ydata = torch.from_numpy(yFull).double()
    priorTemperature = torch.tensor([discretization, discretization, 1.0])  # ?
    pointwise_optimizer = torch.optim.Adam([pointwise_xlatent_torch, pointwise_theta_torch, sigma_torch] + time_constant_param_ls, lr=1e-4)  # , weight_decay = 1.0
    pointwise_lr_scheduler = torch.optim.lr_scheduler.StepLR(pointwise_optimizer, step_size=10000, gamma=0.5)
    cur_loss = 1e12
    for epoch in range(n_iter[1]):
        pointwise_optimizer.zero_grad()
        # compute loss function
        llik = pointwisethetasigmallikTorch(pointwise_xlatent_torch, pointwise_theta_torch, time_constant_param_ls, sigma_torch,
                                            TV_theta_mean, ydata, CovAllDimensionsPyList, fOdeTorch,
                                            priorTemperature, obs_per_day)
        new_loss = -llik
        if epoch % 200 == 0:
            print(epoch, new_loss.item())
            diff = new_loss.item() - cur_loss
            if torch.isnan(new_loss) == False and diff > -0.01 and diff < 0.01:
                break
            cur_loss = new_loss.item()
        new_loss.backward()
        pointwise_optimizer.step()
        pointwise_lr_scheduler.step()


   # Kernel estimation ########################################
    
    pointwise_theta = pointwise_theta_torch.detach().numpy()
    hyperparamList = []
    # optimize the hyperparameters of kernels
    for thetaEachDimension in range(pointwise_theta.shape[1]):
        phi_1 = torch.tensor(phi1_lb_ls[thetaEachDimension], requires_grad=True, dtype=torch.double)
        phi_2 = torch.tensor(phi2_lb_ls[thetaEachDimension], requires_grad=True, dtype=torch.double)
        sigma = torch.tensor(sigma_lb_ls[thetaEachDimension], requires_grad=True, dtype=torch.double)
        kernel_optimizer = torch.optim.Adam([phi_1, phi_2, sigma], lr=learning_rate[1])
        for epoch in range(n_iter[2]):
            kernel_optimizer.zero_grad()
            loss = -kernelllik(phi_1, 
                               phi_2, 
                               sigma, 
                               pointwise_theta_torch[:, thetaEachDimension] - torch.mean(pointwise_theta_torch[:, thetaEachDimension]), 
                               d_matrix, 
                               phi1_lb_ls[thetaEachDimension], 
                               phi2_lb_ls[thetaEachDimension],
                               sigma_lb_ls[thetaEachDimension])
            loss.backward()
            kernel_optimizer.step()
            if epoch % 100 == 0:
                print(epoch, loss.item())
        print(phi_1.detach().item(), phi_2.detach().item(), sigma.detach().item()) 
        hyperparamList.append([phi_1.detach().item(), phi_2.detach().item(), sigma.detach().item()])

    KthetaList = []
    KinvthetaList = []
    for thetaEachDimension in range(pointwise_theta.shape[1]):
        ker = MaternKernel(d_matrix, torch.tensor(hyperparamList[thetaEachDimension][0]), torch.tensor(hyperparamList[thetaEachDimension][1]))
        KthetaList.append(ker)
        KinvthetaList.append(torch.inverse(ker))
                                
# TVMAGI optimization #############################

    TVMAGI_xlatent_torch = torch.tensor(pointwise_xlatent_torch.detach().numpy(), requires_grad=True, dtype=torch.double)
    TVMAGI_theta_torch = torch.tensor(pointwise_theta_torch.detach().numpy(), requires_grad=True, dtype=torch.double)
    TVMAGI_sigma_torch = torch.tensor(inferred_sigma, requires_grad=True, dtype=torch.double)
    TVMAGI_optimizer = torch.optim.Adam([TVMAGI_xlatent_torch, TVMAGI_theta_torch, TVMAGI_sigma_torch] + time_constant_param_ls, lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(TVMAGI_optimizer, step_size=10000, gamma=0.9)
    cur_loss = 1e12
    for epoch in range(n_iter[3]):
        TVMAGI_optimizer.zero_grad()
        # compute loss function
        llik = xthetasigmallikTorch(TVMAGI_xlatent_torch, TVMAGI_theta_torch, time_constant_param_ls, TVMAGI_sigma_torch,
                                                    TV_theta_mean, ydata, CovAllDimensionsPyList, fOdeTorch,
                                                    priorTemperature, KinvthetaList)
        loss = -llik
        if epoch % 500 == 0:
            print(epoch, loss.item())
            if torch.isnan(loss) == False and loss - cur_loss > -0.01 and loss - cur_loss < 0.01:
                break
            cur_loss = loss.item()
        loss.backward()
        TVMAGI_optimizer.step()
        lr_scheduler.step()

