from mean_MAGI import *
from system_derivative import *

torch.set_default_dtype(torch.double)
def pointwisethetasigmallikTorch(xlatent, theta, time_constant_param_ls, sigma, ydata,
                                 CovAllDimensionsPyList, fOdeTorch,
                                 priorTemperature, force_positive, penalty=1e6):
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
        tmp = torch.sparse.mm(CovAllDimensionsPyList[vEachDim]['mphi'], xlatent[:, vEachDim].reshape(-1, 1))
        fitDerivError[:, vEachDim] -= tmp[:, 0]
        nobs[vEachDim] = torch.sum(torch.isfinite(ydata[:, vEachDim]))
        obsIdx = torch.isfinite(ydata[:, vEachDim])
        fitLevelErrorSumSq[vEachDim] = torch.sum(torch.square(xlatent[obsIdx, vEachDim] - ydata[obsIdx, vEachDim]))
    res[:, 0] = -0.5 * fitLevelErrorSumSq / sigmaSq - torch.log(sigma + 0.001) * nobs
    res[:, 0] /= priorTemperature[2]
    KinvfitDerivError = torch.zeros([n, pdimension]).double()
    CinvX = torch.zeros([n, pdimension]).double()
    for vEachDim in range(pdimension):
        # inverse of K
        KinvfitDerivError[:, vEachDim] = torch.sparse.mm(CovAllDimensionsPyList[vEachDim]['Kinv'],
                                                         fitDerivError[:, vEachDim].reshape(-1, 1))[:, 0]
        # inverse of Cd
        CinvX[:, vEachDim] = torch.sparse.mm(CovAllDimensionsPyList[vEachDim]['Cinv'],
                                             xlatent[:, vEachDim].reshape(-1, 1))[:, 0]
    res[:, 1] = -0.5 * torch.sum(fitDerivError * KinvfitDerivError, dim=0) / priorTemperature[0]
    #  prior distriobution of X
    res[:, 2] = -0.5 * torch.sum(xlatent * CinvX, dim=0) / priorTemperature[1]
    if force_positive is None:
        return torch.sum(res)
    elif len(force_positive) != thetadimension:
        raise ValueError('The dimension of force_positive should equal parameter dimension!')
    for EachDim in range(thetadimension):
        output = torch.sum(res)
        if force_positive[EachDim]:
            output -= penalty * torch.sum(torch.square(theta[:, EachDim] - torch.clamp(theta[:, EachDim], min=0.)))
    return output

class pointwise_model():
    def __init__(self, ODE_dynamic, result, force_positive=None, penalty=1e6, use_trajectory = 'inferred'):
        inferred_theta = result[0] 
        inferred_sigma = result[1]
        inferred_trajectory = result[2]
        CovAllDimensionsPyList = result[3]
        discretization = ODE_dynamic.yFull.shape[0] / ODE_dynamic.yobs.shape[0]
        priorTemperature = torch.tensor([discretization, discretization, 1.0])  # ?
        
        if use_trajectory == 'observation':
            pointwise_xlatent_torch = torch.tensor(xInitExogenous, requires_grad=True, dtype=torch.double)
        elif use_trajectory == 'inferred':
            pointwise_xlatent_torch = torch.tensor(inferred_trajectory.transpose(), requires_grad=True, dtype=torch.double)
        else:
            raise ValueError("Use_trajectory must be either 'obervation' or 'inferred'!")
        TV_theta_mean = np.zeros(int(sum(ODE_dynamic.is_time_varying)))
        tv_index = 0
        for thetaEachDim in range(ODE_dynamic.theta_dim):
            if ODE_dynamic.is_time_varying[thetaEachDim] == True:
                TV_theta_mean[tv_index] = inferred_theta[thetaEachDim]
                tv_index += 1
        tmp1 = np.array([TV_theta_mean])
        initial_tvtheta = np.repeat(tmp1, pointwise_xlatent_torch.shape[0], axis=0)
        
        time_constant_param_ls = []
        for thetaEachDim in range(ODE_dynamic.theta_dim):
            if ODE_dynamic.is_time_varying[thetaEachDim] == 0:
                param_name = ODE_dynamic.param_names[thetaEachDim]
                locals()[param_name] = torch.tensor([inferred_theta[thetaEachDim]], requires_grad=True, dtype=torch.double)
                time_constant_param_ls.append(eval(param_name))
        
        
        self.CovAllDimensionsPyList = CovAllDimensionsPyList
        self.fOdeTorch = ODE_dynamic.fOdeTorch
        self.inferred_sigma = inferred_sigma
        self.inferred_theta = inferred_theta  
        self.inferred_trajectory = inferred_trajectory        
        self.ODE_dynamic = ODE_dynamic
        self.param_names = ODE_dynamic.param_names
        self.pointwise_sigma_torch = torch.tensor(inferred_sigma, requires_grad=True, dtype=torch.double)
        self.pointwise_theta_torch = torch.tensor(initial_tvtheta, requires_grad=True, dtype=torch.double)
        self.pointwise_xlatent_torch = pointwise_xlatent_torch
        self.priorTemperature = priorTemperature
        self.time_constant_param_ls = time_constant_param_ls
        self.tvecFull = ODE_dynamic.tvecFull
        self.TV_theta_mean = TV_theta_mean
        self.ydata = torch.from_numpy(ODE_dynamic.yFull).double()
        self.yobs = torch.from_numpy(ODE_dynamic.yobs).double()
        self.force_positive = force_positive
        self.penalty = penalty
    def train(self, lr = 1e-4, niter = 50000):
        
        
        pointwise_optimizer = torch.optim.Adam(
            [self.pointwise_xlatent_torch, self.pointwise_theta_torch, self.pointwise_sigma_torch] + self.time_constant_param_ls,
            lr=lr)  # , weight_decay = 1.0
        pointwise_lr_scheduler = torch.optim.lr_scheduler.StepLR(pointwise_optimizer, step_size=10000, gamma=0.5)
        cur_loss = np.Inf
        for epoch in range(niter):
            pointwise_optimizer.zero_grad()
            # compute loss function
            llik = pointwisethetasigmallikTorch(self.pointwise_xlatent_torch, self.pointwise_theta_torch, self.time_constant_param_ls,
                                                self.pointwise_sigma_torch,
                                                self.ydata, self.CovAllDimensionsPyList, self.ODE_dynamic.fOdeTorch,
                                                self.priorTemperature, self.force_positive, self.penalty)
            new_loss = -llik
                
            if epoch % 200 == 0:
                if epoch % 1000 == 0:
                    print('%d/%d iteration: %.6f' %(epoch+1,niter,new_loss.item()))
                diff = new_loss.item() - cur_loss
                if torch.isnan(new_loss) == False and diff > -0.1 and diff < 0.1:
                    print('Optimization converges')
                    break
                cur_loss = new_loss.item()
            new_loss.backward()
            pointwise_optimizer.step()
            pointwise_lr_scheduler.step()
        
    







