import scipy.special as fun
import torch
import numpy as np

torch.set_default_dtype(torch.double)
class Bessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        ctx._nu = nu
        ctx.save_for_backward(inp)
        mat = fun.kv(nu, inp.detach().numpy())
        return (torch.from_numpy(np.array(mat)))

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        nu = ctx._nu
        grad_in = grad_out.numpy() * np.array(fun.kvp(nu, inp.detach().numpy()))
        return (torch.from_numpy(grad_in), None)


class generalMatern(object):

    # has_lengthscale = True

    def __init__(self, nu, lengthscale, **kwargs):
        # super(Matern,self).__init__(**kwargs)
        self.nu = nu
        self.log_lengthscale = torch.tensor(np.log(lengthscale))
        self.log_lengthscale.requires_grad_(True)

    def _set_lengthscale(self, lengthscale):
        self.log_lengthscale = torch.tensor(np.log(lengthscale))

    def lengthscale(self):
        return (torch.exp(self.log_lengthscale).item())

    def forward(self, x1, x2=None, **params):
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if x2 is None: x2 = x1
        r_ = (x1.reshape(-1, 1) - x2.reshape(1, -1)).abs()
        r_ = np.sqrt(2. * self.nu) * r_ / lengthscale
        # handle limit at 0, allows more efficient backprop
        r_ = torch.clamp(r_, min=1e-15)
        C_ = np.power(2, 1 - self.nu) * np.exp(-fun.loggamma(self.nu)) * torch.pow(r_, self.nu)
        mat = Bessel.apply(r_, self.nu)
        C_ = C_ * mat
        return (C_)

    def C(self, x1, x2=None):
        return (self.forward(x1, x2).detach())


def GPTrain(train_x, train_y, nu, lengthscale_lb, learning_rate=1e-6, noisy=True, max_iter=5, verbose=False,
            eps=1e-6):
    # preprocess input data
    n = train_x.size(0)
    # normalized x to 0 and 1
    x_range = [torch.min(train_x).item(), torch.max(train_x).item()]
    train_x = (train_x - x_range[0]) / (x_range[1] - x_range[0])
    #     train_x[0] = eps
    # set up kernel
    kernel = generalMatern(nu=nu, lengthscale=1.1 * lengthscale_lb / (x_range[1] - x_range[0]))
    # lambda = noise/outputscale
    log_lambda = torch.tensor(np.log(1e-2))
    log_lambda.requires_grad_(True)
    loglb_normalized = torch.log(torch.tensor(lengthscale_lb / (x_range[1] - x_range[0])))
    optimizer = torch.optim.LBFGS([kernel.log_lengthscale, log_lambda], lr=learning_rate)
    # training
    prev_loss = np.Inf
    for i in range(max_iter):
        R = kernel.forward(train_x) + torch.exp(log_lambda) * torch.eye(n)
        e, v = torch.eig(R, eigenvectors=True)
        e = e[:, 0]  # eigenvalues
        a = v.T @ torch.ones(n)
        b = v.T @ train_y
        mean = ((a / e).T @ b) / ((a / e).T @ a)
        d = v.T @ (train_y - mean)
        outputscale = 1. / n * (d / e).T @ d

        def closure():
            optimizer.zero_grad()
            R = kernel.forward(train_x) + torch.exp(log_lambda) * torch.eye(n)
            e, v = torch.eig(R, eigenvectors=True)
            e = e[:, 0]  # eigenvalues
            a = v.T @ torch.ones(n)
            b = v.T @ train_y
            mean = ((a / e).T @ b) / ((a / e).T @ a)
            d = v.T @ (train_y - mean)
            outputscale = 1. / n * (d / e).T @ d
            loss = torch.log(outputscale) + torch.mean(torch.log(e))
            tmp0 = torch.clamp(kernel.log_lengthscale, max=0.)
            loss = loss + 1e3 * torch.sum(torch.square(kernel.log_lengthscale - tmp0))
            tmp = torch.clamp(kernel.log_lengthscale, min=loglb_normalized)
            loss = loss + 1e3 * torch.sum(torch.square(kernel.log_lengthscale - tmp))
            tmp2 = torch.clamp(log_lambda, min=np.log(1e-6))
            loss = loss + 1e3 * torch.sum(torch.square(log_lambda - tmp2))
            loss.backward()
            return loss

        optimizer.step(closure)

    R = kernel.forward(train_x) + torch.exp(log_lambda) * torch.eye(n)
    Rinv = torch.inverse(R)
    ones = torch.ones(n)
    mean = ((ones.T @ Rinv @ train_y) / (ones.T @ Rinv @ ones)).item()
    outputscale = (1 / n * (train_y - mean).T @ Rinv @ (train_y - mean)).item()
    noisescale = outputscale * torch.exp(log_lambda).item()
    return mean, outputscale, noisescale, (x_range[1] - x_range[0]) * torch.exp(kernel.log_lengthscale).item()


def to_band(matrix, bandwidth=30):
    dim = matrix.shape[0]
    for i in range(dim):
        for j in range(dim):
            if i > j + bandwidth or i < j - bandwidth:
                matrix[i][j] = 0
    return matrix.to_sparse()
    
class estimate_kernel():
    def __init__(self, pointwise_problem, nu=2.01, lengthscale_lb=3.0):
        KinvthetaList = []
        for EachDim in range(pointwise_problem.pointwise_theta_torch.shape[1]):
            a, outputscale, c, lengthscale = GPTrain(torch.tensor(pointwise_problem.tvecFull),
                                        pointwise_problem.pointwise_theta_torch.detach()[:, EachDim], nu, lengthscale_lb)
            ker = outputscale * generalMatern(nu, lengthscale).C(torch.tensor(pointwise_problem.tvecFull))
            KinvthetaList.append(to_band(torch.inverse(ker)))
        self.KinvthetaList = KinvthetaList
        
