import numpy as np
import torch

def vectorize(xlatent, theta, sigma, time_constant_param_ls):
        t1 = torch.reshape(xlatent.detach(), (-1,))
        t2 = torch.reshape(theta.detach(), (-1,))
        t3 = torch.reshape(sigma.detach(), (-1,))
        long_vec = torch.cat((t1, t2, t3))
        for i in range(len(time_constant_param_ls)):
            long_vec = torch.cat((long_vec, time_constant_param_ls[i].detach()))
        return long_vec

def get_dim(tensor_shape):
    if len(tensor_shape) == 0:
        return 1
    if len(tensor_shape) == 1:
        return tensor_shape[0]
    dim = 1
    for i in range(len(tensor_shape)):
        dim *= tensor_shape[i]
    return dim

def devectorize(long_tensor, xlatent_shape, theta_shape, sigma_shape, time_constant_param_dim):
    x_latent_dim = get_dim(xlatent_shape)
    theta_dim = get_dim(theta_shape)
    sigma_dim = get_dim(sigma_shape)
    time_constant_param_ls = []
    xlatent = torch.reshape(long_tensor[:x_latent_dim], xlatent_shape)
    theta = torch.reshape(long_tensor[x_latent_dim:x_latent_dim + theta_dim], theta_shape)
    sigma = torch.reshape(long_tensor[x_latent_dim + theta_dim:x_latent_dim + theta_dim + sigma_dim], sigma_shape)
    for each in range(x_latent_dim + theta_dim + sigma_dim, long_tensor.shape[0]):
        time_constant_param_ls.append(torch.tensor([long_tensor[each]]))
    return xlatent, theta, sigma, time_constant_param_ls

def NegLogLikelihood(xlatent, theta, sigma, time_constant_param_ls,
                     inferred_theta=inferred_theta,
                     ydata=ydata,
                     CovAllDimensionsPyList=CovAllDimensionsPyList,
                     fOdeTorch=fOdeTorch,
                     priorTemperature=priorTemperature,
                     KinvthetaList=KinvthetaList):
    # length of observed y (t)
    n = ydata.shape[0]
    pdimension = ydata.shape[1]
    thetadimension = theta.shape[1]
    sigmaSq = torch.pow(sigma, 2)
    fderiv = fOdeTorch(theta, xlatent, time_constant_param_ls)
    res = torch.zeros([pdimension, 3]).double()
    res_theta = torch.zeros(thetadimension).double()
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
                        theta[:, thetaEachDim] - inferred_theta[thetaEachDim]).reshape(-1, 1))[:, 0])
    res[:, 1] = -0.5 * torch.sum(fitDerivError * KinvfitDerivError, dim=0) / priorTemperature[0]
    res[:, 2] = -0.5 * torch.sum(xlatent * CinvX, dim=0) / priorTemperature[1]

    return -(torch.sum(res) + torch.sum(res_theta))

class HMC:
    def __init__(self, negllik, all_theta, xlatent_shape, theta_shape, sigma_shape, time_constant_param_ls,
                 lsteps=100, epsilon=1e-5, n_samples=16000, upper_bound=None, lower_bound=None, burn_in_ratio=0.5):
        self.all_theta = all_theta
        self.theta_shape = theta_shape
        self.xlatent_shape = xlatent_shape
        self.sigma_shape = sigma_shape
        self.constant_dim = len(time_constant_param_ls)
        self.lsteps = lsteps
        self.epsilon = epsilon * torch.ones(all_theta.shape)
        self.burn_in_ratio = burn_in_ratio
        self.n_samples = n_samples
        self.total_samples = int(n_samples / (1 - burn_in_ratio))
        self.NegLogLikelihood = negllik
        self.ub = upper_bound
        if upper_bound is not None:
            if upper_bound.shape[0] != all_theta.shape[0]:
                raise ValueError
        self.lb = lower_bound
        if lower_bound is not None:
            if lower_bound.shape[0] != all_theta.shape[0]:
                raise ValueError

    def NegLogLikelihood_vec(self, all_theta):
        xlatent_0, theta_0, sigma_0, constant_param_ls_0 = devectorize(all_theta, self.xlatent_shape,
                                                                       self.theta_shape, self.sigma_shape,
                                                                       self.constant_dim)
        return NegLogLikelihood(xlatent_0, theta_0, sigma_0, constant_param_ls_0)

    def Nabla(self, theta_torch):
        theta_torch = theta_torch.detach()
        xlatent, theta, sigma, constant_param_ls = devectorize(theta_torch, self.xlatent_shape, self.theta_shape,
                                                               self.sigma_shape, self.constant_dim)
        xlatent.requires_grad = True
        theta.requires_grad = True
        sigma.requires_grad = True
        for each in constant_param_ls:
            each.requires_grad = True
        llik = self.NegLogLikelihood(xlatent, theta, sigma, constant_param_ls)
        llik.backward()
        constant_param_deriv_ls = []
        for each in constant_param_ls:
            constant_param_deriv_ls.append(each.grad)
        v = vectorize(xlatent.grad, theta.grad, sigma.grad, constant_param_deriv_ls)

        return v

    def sample(self, all_theta, TV_theta_mean, ydata, CovAllDimensionsPyList, fOdeTorch, priorTemperature,
               KinvthetaList):
        def bounce(m, lb, ub):
            if lb is None and ub is None:
                return m
            if lb is None:
                max_tensor = torch.clamp(m - ub, min=0)
                return m - 2 * max_tensor
            if ub is None:
                min_tensor = torch.clamp(lb - m, min=0)
                return m + 2 * min_tensor
            if torch.sum(lb < ub) < m.shape[0]:
                raise ValueError
            if torch.sum(m >= lb) == m.shape[0] and torch.sum(m <= ub) == m.shape[0]:
                return m
            if torch.sum(m >= lb) < m.shape[0]:
                min_tensor = torch.clamp(lb - m, min=0)
                return bounce(m + 2 * min_tensor, lb, ub)
            if torch.sum(m <= ub) < m.shape[0]:
                max_tensor = torch.clamp(m - ub, min=0)
                return bounce(m - 2 * max_tensor, lb, ub)

        trace_val = np.zeros(self.total_samples)
        samples = np.zeros((self.total_samples, self.all_theta.shape[0]))
        random_ls = np.random.uniform(0, 1, self.total_samples)
        acceptance_ls = np.zeros(self.total_samples)
        nan_ls = np.zeros(self.total_samples)
        cur_theta = self.all_theta.clone().detach()
        for EachIter in range(self.total_samples):  ############
            cur_nllik_1 = self.NegLogLikelihood_vec(cur_theta).detach()
            rstep = torch.rand(self.epsilon.shape) * self.epsilon + self.epsilon
            p = torch.normal(mean=0., std=torch.ones(self.all_theta.shape))
            cur_p = p.clone()
            theta = cur_theta.clone()
            p = p - rstep * self.Nabla(theta).clone() / 2
            for i in range(self.lsteps):
                theta = theta + rstep * p
                nabla_torch = self.Nabla(theta).clone()
                p = p - rstep * nabla_torch
                theta = bounce(theta, self.lb, self.ub)

            p = p - rstep * self.Nabla(theta).clone() / 2

            new_nllik = self.NegLogLikelihood_vec(theta)
            new_p = 0.5 * torch.sum(torch.square(p))
            new_H = new_nllik + new_p
            cur_nllik = self.NegLogLikelihood_vec(cur_theta).detach()
            cur_H = cur_nllik + 0.5 * torch.sum(torch.square(cur_p))
            #             print(new_H, cur_H)

            if torch.isnan(theta[0]) or torch.isnan(new_H):
                samples[EachIter] = cur_theta.clone()
                nan_ls[EachIter] = 1
                self.epsilon *= 0.9
                print('NaN!')
            else:
                # accept
                tmp = float(torch.exp(cur_H - new_H))
                #                 print(tmp)
                if tmp > random_ls[EachIter]:
                    samples[EachIter] = theta.clone()
                    cur_theta = theta.clone()
                    acceptance_ls[EachIter] = 1
                # reject
                else:
                    samples[EachIter] = cur_theta.clone()

            trace_val[EachIter] = self.NegLogLikelihood_vec(cur_theta).item()

            if EachIter > 200 and EachIter < self.total_samples - self.n_samples:
                if np.sum(acceptance_ls[EachIter - 100: EachIter]) < 60:
                    # decrease epsilon
                    self.epsilon *= 0.995
                if np.sum(acceptance_ls[EachIter - 100: EachIter]) > 90:
                    # increase epsilon
                    self.epsilon *= 1.005
            if EachIter % 100 == 0 and EachIter > 100:
                print(EachIter)
                print(cur_nllik)
                print('acceptance rate: ', np.sum(acceptance_ls[EachIter - 100: EachIter]) / 100)
                if EachIter < self.total_samples - self.n_samples:
                    standard_deviation = torch.tensor(np.std(samples[EachIter - 100:EachIter, :], axis=0))
                    if torch.mean(standard_deviation) > 1e-6:
                        self.epsilon = 0.05 * standard_deviation * torch.mean(self.epsilon) / torch.mean(
                            standard_deviation) + 0.95 * self.epsilon
        return samples, acceptance_ls, trace_val, nan_ls  # [self.total_samples-self.n_samples:, :]

all_theta_TVMAGI = vectorize(TVMAGI_xlatent_torch, TVMAGI_theta_torch, TVMAGI_sigma_torch, time_constant_param_ls)
all_theta_pointwise = vectorize(pointwise_xlatent_torch, pointwise_theta_torch, TVMAGI_sigma_torch,
                                time_constant_param_ls)
sampler = HMC(NegLogLikelihood, all_theta_TVMAGI,
              pointwise_xlatent_torch.shape,
              pointwise_theta_torch.shape,
              TVMAGI_sigma_torch.shape,
              time_constant_param_ls,
              lower_bound=torch.zeros(all_theta_pointwise.shape))
# sampler.Nabla(all_theta)
# lower_bound = torch.zeros(all_theta_pointwise.shape)
samples, b, c, d = sampler.sample(all_theta_TVMAGI, TV_theta_mean, ydata, CovAllDimensionsPyList, fOdeTorch,
                                  priorTemperature, KinvthetaList)
k = samples[8000:, 256:256 + 192]
days = 32
obs_per_day = 2
beta_ls = np.zeros((8000, 64))
ve_ls = np.zeros((8000, 64))
pd_ls = np.zeros((8000, 64))
vi_ls = samples[8000:, 448:458]
xinit_ls = samples[8000:, :4]
for i in range(8000):
    val = np.zeros((64, 3))
    for j in range(64):
        val[j] = k[i].reshape(-1, 3)[j]
    beta_ls[i] = val[:, 0]
    ve_ls[i] = val[:, 1]
    pd_ls[i] = val[:, 2] / 4
