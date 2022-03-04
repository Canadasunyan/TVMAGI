import numpy as np
import pickle
import torch
import scipy
import argparse
import sys
from arma import ode_system, solve_magi
from arma import matrix
from system_derivative import *

torch.set_default_dtype(torch.double)
class mean_model():
    def __init__(self, ODE_dynamic,
                sigmaExogenous=np.array([]),
                phiExogenous=np.array([[]]),
                thetaInitExogenous=np.array([]),
                muExogenous=np.array([[]]),
                dotmuExogenous=np.array([[]]),
                priorTemperatureObs=1.0,
                kernel="generalMatern",
                nstepsHmc=100,
                burninRatioHmc=0.5,
                niterHmc=15001,
                stepSizeFactorHmc=0.01,
                nEpoch=1,
                bandSize=30,
                useFrequencyBasedPrior=True,
                useBand=False,
                useMean=False,
                useScalerSigma=False,
                useFixedSigma=False,
                verbose=True):
        
        self.yobs = ODE_dynamic.yobs
        self.dynamic = ODE_dynamic
        self.Ode_system = ode_system("ODE-python", ODE_dynamic.fOde, ODE_dynamic.fOdeDx, ODE_dynamic.fOdeDtheta,
                                thetaLowerBound=ODE_dynamic.theta_lowerbound,
                                thetaUpperBound=ODE_dynamic.theta_upperbound)
        self.theta_dim = ODE_dynamic.theta_dim 
        self.yFull = ODE_dynamic.yFull
        self.tvecObs = ODE_dynamic.tvecObs
        self.tvecFull = ODE_dynamic.tvecFull
        self.sigmaExogenous = sigmaExogenous
        self.phiExogenous = phiExogenous
        self.xInitExogenous = ODE_dynamic.xInitExogenous
        self.thetaInitExogenous = thetaInitExogenous
        self.muExogenous = muExogenous
        self.dotmuExogenous = dotmuExogenous
        self.priorTemperatureLevel = ODE_dynamic.yFull.shape[0] / ODE_dynamic.yobs.shape[0]
        self.priorTemperatureDeriv = ODE_dynamic.yFull.shape[0] / ODE_dynamic.yobs.shape[0]
        self.priorTemperatureObs = priorTemperatureObs
        self.kernel = kernel
        self.nstepsHmc = nstepsHmc
        self.burninRatioHmc = burninRatioHmc
        self.niterHmc = niterHmc
        self.stepSizeFactorHmc = stepSizeFactorHmc
        self.nEpoch = nEpoch
        self.bandSize = bandSize
        self.useFrequencyBasedPrior = useFrequencyBasedPrior
        self.useBand = useBand
        self.useMean = useMean
        self.useScalerSigma = useScalerSigma
        self.useFixedSigma = useFixedSigma
        self.verbose = verbose
    
    def to_band(self, matrix, bandwidth=30):
        dim = matrix.shape[0]
        for i in range(dim):
            for j in range(dim):
                if i > j + bandwidth or i < j - bandwidth:
                    matrix[i][j] = 0
        return matrix.to_sparse()

    def solve(self):    
        result = solve_magi(
                self.yFull,
                self.Ode_system,
                self.tvecFull,
                self.sigmaExogenous,
                self.phiExogenous,
                self.xInitExogenous,
                self.thetaInitExogenous,
                self.muExogenous,
                self.dotmuExogenous,
                self.priorTemperatureLevel,
                self.priorTemperatureDeriv,
                self.priorTemperatureObs,
                self.kernel,
                self.nstepsHmc,
                self.burninRatioHmc,
                self.niterHmc,
                self.stepSizeFactorHmc,
                self.nEpoch,
                self.bandSize,
                self.useFrequencyBasedPrior,
                self.useBand,
                self.useMean,
                self.useScalerSigma,
                self.useFixedSigma,
                self.verbose)
        samplesCpp = result['samplesCpp']
        llikId = 0
        xId = range(np.max(llikId) + 1, np.max(llikId) + self.yFull.size + 1)
        # dimension of theta
        thetaId = range(np.max(xId) + 1, np.max(xId) + self.theta_dim + 1)
        sigmaId = range(np.max(thetaId) + 1, np.max(thetaId) + self.yFull.shape[1] + 1)
        burnin = int(self.niterHmc * 0.5)
        xsampled = samplesCpp[xId, (burnin + 1):]
        xsampled = xsampled.reshape([self.yFull.shape[1], self.yFull.shape[0], -1])
        CovAllDimensionsPyList = []
        thetaSampled = samplesCpp[thetaId, (burnin + 1):]
        inferred_theta = np.mean(thetaSampled, axis=-1)
#             print(inferred_theta)
        priorTemperatureLevel=self.yFull.shape[0] / self.yobs.shape[0]
        priorTemperatureDeriv=self.yFull.shape[0] / self.yobs.shape[0]
        sigmaSampled = samplesCpp[sigmaId, (burnin + 1):]
        inferred_sigma = np.mean(sigmaSampled, axis=-1)
        inferred_trajectory = np.mean(xsampled, axis=-1)
        for each_gpcov in result['result_solved'].covAllDimensions:
            each_pycov = dict(
                Cinv=self.to_band(torch.from_numpy(matrix(each_gpcov.Cinv)).double(), bandwidth=self.bandSize),
                Kinv=self.to_band(torch.from_numpy(matrix(each_gpcov.Kinv)).double(), bandwidth=self.bandSize),
                mphi=self.to_band(torch.from_numpy(matrix(each_gpcov.mphi)).double(), bandwidth=self.bandSize),
            )
            CovAllDimensionsPyList.append(each_pycov)
        return inferred_theta, inferred_sigma, inferred_trajectory, CovAllDimensionsPyList, self.yFull
    
    