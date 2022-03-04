import numpy as np
import torch

'''
This module describes the dynamic of the system, including calculation of the derivatives.
'''

torch.set_default_dtype(torch.double)
class system_dynamic():
    def __init__(self, yobs, is_time_varying, tvecObs=None, tvecFull=None, param_names=None,
                theta_lowerbound=None,
                theta_upperbound=None, **kwargs):       
        theta_dim = len(is_time_varying)
        
        if param_names is None:
            param_names = ['theta_' + str(i) for i in range(len(is_time_varying))]
        if len(param_names) != len(is_time_varying):
            raise ValueError('The dimension of param_names and is_time_varying must be equal!')        
        
        if tvecObs is None:
            tvecObs = np.arange(0, yobs.shape[0], 1)        
        if yobs.shape[0] != tvecObs.shape[0]:
            raise ValueError('The length of tvecObs and yobs do not match!')
        if tvecFull is None:
            tvecFull = tvecObs
        yFull = np.ndarray([tvecFull.shape[0], yobs.shape[1]])
        yFull.fill(np.nan)
        idxFull = 0
        for idxObs in range(yobs.shape[0]): 
            while np.abs(tvecFull[idxFull]-tvecObs[idxObs]) > 0.0001:
                idxFull += 1
                if idxFull >= yFull.shape[0]:
                    raise ValueError('tvecObs must be a subarray of tvecFull!')
                    break
            yFull[idxFull] = yobs[idxObs] 
          
        xInitExogenous = np.zeros_like(yFull)
        nobs, p_dim = yobs.shape[0], yobs.shape[1]
        for i in range(p_dim):
            xInitExogenous[:, i] = np.interp(tvecFull, tvecObs, yobs[:, i])
            
        if theta_lowerbound is None:
            theta_lowerbound = -np.inf * np.ones(theta_dim)
        if theta_upperbound is None:
            theta_upperbound = np.inf * np.ones(theta_dim)
        if theta_upperbound.shape[0] != theta_lowerbound.shape[0]:
            raise ValueError('The dimensions of upper bound and lower bound do not match!')
            
        self.yobs = yobs
        self.is_time_varying = is_time_varying
        self.tvecObs = tvecObs
        self.tvecFull = tvecFull 
        self.param_names = param_names                    
        self.theta_dim = theta_dim             
        self.yFull = yFull  
        self.xInitExogenous = xInitExogenous
        self.theta_lowerbound = theta_lowerbound
        self.theta_upperbound = theta_upperbound
        for key, value in kwargs.items():
            setattr(self, key, value)
    # Derivatves of X according to the ODE structure
    def fOde(self, theta, x):
        """
        theta: list[4]: beta, ve, vi, pd
        x: array(n, 4)
        r: array(n, 2)
        """
        logS = x[:, 0]
        logE = x[:, 1]
        logI = x[:, 2]
        logD = x[:, 3]
        logSdt = -theta[0] * np.exp(logI) / self.N  # (1)
        logEdt = theta[0] * np.exp(logS + logI - logE) / self.N - theta[1]  # (2)
        logIdt = np.exp(logE - logI) * theta[1] - theta[2]  # (3)
        logDdt = np.exp(logI - logD) * 0.25 * theta[3] * theta[2]  # (4)
        return np.stack([logSdt, logEdt, logIdt, logDdt], axis=1)

    # Derivatives of X
    def fOdeDx(self, theta, x):
        """
        returns derivation of x given theta
        theta: list[4]
        x: array(n, 4)
        r: array(n, 4, 4)
        """
        resultDx = np.zeros(shape=[np.shape(x)[0], np.shape(x)[1], np.shape(x)[1]])
        logS = x[:, 0]
        logE = x[:, 1]
        logI = x[:, 2]
        logD = x[:, 3]
        # [:, i, j]: derivative of ith component in jth equation
        # (1) / dI
        resultDx[:, 2, 0] = -theta[0] * np.exp(logI) / self.N
        # (1) / dS, (1) /dE, (1) / dD = 0
        # (2) / dS
        resultDx[:, 0, 1] = theta[0] * np.exp(logS + logI - logE) / self.N
        # (2) / dE
        resultDx[:, 1, 1] = -theta[0] * np.exp(logS + logI - logE) / self.N
        # (2) / dI
        resultDx[:, 2, 1] = theta[0] * np.exp(logS + logI - logE) / self.N
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

    def fOdeDtheta(self, theta, x):
        """
        returns derivation of theta given x
        theta: list[4]
        x: array(n, 4)
        r: array(n, 4, 4)
        """
        resultDtheta = np.zeros(shape=[np.shape(x)[0], np.shape(theta)[0], np.shape(x)[1]])
        logS = x[:, 0]
        logE = x[:, 1]
        logI = x[:, 2]
        logD = x[:, 3]
        # (1) / dRe
        resultDtheta[:, 0, 0] = -np.exp(logI) / self.N
        # (2) / d theta[0]
        resultDtheta[:, 0, 1] = np.exp(logS + logI - logE) / self.N
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

    def fOdeTorch(self, theta, x, constant_param_ls):
        """
        theta: list[4]: beta, ve, vi, pd
        x: array(n, 4)
        r: array(n, 2)
        """
        logS = x[:, 0]
        logE = x[:, 1]
        logI = x[:, 2]
        logD = x[:, 3]
        logSdt = -theta[:, 0] * torch.exp(logI) / self.N  # (1)
        logEdt = theta[:, 0] * torch.exp(logS + logI - logE) / self.N - theta[:, 1]  # (2)
        logIdt = torch.exp(logE - logI) * theta[:, 1] - constant_param_ls[0]  # (3)
        # reparametrize on pd
        logDdt = torch.exp(logI - logD) * 0.25 * theta[:, 2] * constant_param_ls[0]  # (4)
        return torch.stack([logSdt, logEdt, logIdt, logDdt], axis=1)