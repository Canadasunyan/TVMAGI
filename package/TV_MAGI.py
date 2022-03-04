import torch
import numpy as np

def xthetasigmallikTorch(xlatent, theta, time_constant_param_ls, sigma, inferred_theta, ydata, CovAllDimensionsPyList,
                         fOdeTorch,
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
        fitDerivError[:, vEachDim] -= torch.sparse.mm(CovAllDimensionsPyList[vEachDim]['mphi'],
                                                      xlatent[:, vEachDim].reshape(-1, 1))[:, 0]
        nobs[vEachDim] = torch.sum(torch.isfinite(ydata[:, vEachDim]))
        obsIdx = torch.isfinite(ydata[:, vEachDim])
        fitLevelErrorSumSq[vEachDim] = torch.sum(torch.square(xlatent[obsIdx, vEachDim] - ydata[obsIdx, vEachDim]))
    res[:, 0] = -0.5 * fitLevelErrorSumSq / sigmaSq - torch.log(sigma + 0.0001) * nobs
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
    for thetaEachDim in range(thetadimension):
        res_theta[thetaEachDim] = -0.5 * torch.sum(
            (theta[:, thetaEachDim] - inferred_theta[thetaEachDim]) @ torch.sparse.mm(KinvthetaList[thetaEachDim], (
                    theta[:, thetaEachDim] - inferred_theta[thetaEachDim]).reshape(-1, 1)))
    res[:, 1] = -0.5 * torch.sum(fitDerivError * KinvfitDerivError, dim=0) / priorTemperature[0]
    res[:, 2] = -0.5 * torch.sum(xlatent * CinvX, dim=0) / priorTemperature[1]
    theta_lb = torch.clamp(theta[:, 2], min=0.)
    return torch.sum(res) + torch.sum(res_theta)


class TVMAGI_model():
    def __init__(self, pointwise_system, KinvthetaList):
        self.ODE_dynamic = pointwise_system.ODE_dynamic
        self.fOdeTorch = pointwise_system.fOdeTorch
        self.KinvthetaList = KinvthetaList
        self.TVMAGI_xlatent_torch = torch.tensor(pointwise_system.pointwise_xlatent_torch.detach().numpy(), requires_grad=True,
                                        dtype=torch.double)
        self.TVMAGI_theta_torch = torch.tensor(pointwise_system.pointwise_theta_torch.detach().numpy(), requires_grad=True, dtype=torch.double)
        self.TVMAGI_sigma_torch = torch.tensor(pointwise_system.inferred_sigma, requires_grad=True, dtype=torch.double)
        self.time_constant_param_ls = pointwise_system.time_constant_param_ls.copy()
        self.TV_theta_mean = pointwise_system.TV_theta_mean
        self.CovAllDimensionsPyList = pointwise_system.CovAllDimensionsPyList
        self.priorTemperature = pointwise_system.priorTemperature
        self.ydata = pointwise_system.ydata
    def train(self, lr=1e-5, nepoch=50000):
        TVMAGI_optimizer = torch.optim.Adam(
            [self.TVMAGI_xlatent_torch, self.TVMAGI_theta_torch, self.TVMAGI_sigma_torch] + self.time_constant_param_ls, lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(TVMAGI_optimizer, step_size=5000, gamma=0.8)
        cur_loss = np.Inf
        for epoch in range(nepoch):
            TVMAGI_optimizer.zero_grad()
            # compute loss function
            llik = xthetasigmallikTorch(self.TVMAGI_xlatent_torch, self.TVMAGI_theta_torch, self.time_constant_param_ls,
                                        self.TVMAGI_sigma_torch,
                                        self.TV_theta_mean, self.ydata, self.CovAllDimensionsPyList, self.fOdeTorch,
                                        self.priorTemperature, self.KinvthetaList)
            new_loss = -llik
            if epoch % 200 == 0:
                if epoch % 1000 == 0:
                    print('%d/%d iteration: %.6f' %(epoch+1,nepoch,new_loss.item()))
                diff = new_loss.item() - cur_loss
                if torch.isnan(new_loss) == False and diff > -0.1 and diff < 0.1:
                    break
                cur_loss = new_loss.item()
            new_loss.backward()
            TVMAGI_optimizer.step()
            lr_scheduler.step()
    
    
    
    
    
    