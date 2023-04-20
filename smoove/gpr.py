from functools import partial
import numpy as np
import numba
from scipy.optimize import fmin_l_bfgs_b as fmin
from smoove.utils import abs_diff, diag_dot
from smoove.kernels.mattern52 import mat52
from scipy.special import polygamma
from smoove.utils import nufunc

def dZdtheta(theta, y, Sigma, kernel, xx):
    '''
    Return log-marginal likelihood and derivs.

    Inputs:
        theta       - k-length array of hyper-parameters with noise scaling as last parameter
        y           - N-length vector of data
        Sigma       - N-length vector with noise covariance
        kernel      - function that only takes hyper-parameters as inputs and returns the kernel as well as derivs

    Outputs:
        Z           - the negative log of the marginal likelihood (evidence)
        dZ          - derivs of Z w.r.t. theta

    Note that the function kgrad must take theta[0:-1] as the only input
    and must return the value and gradient in terms of a scalar and an
    array of
    Use eg. partial
    '''
    N = y.size
    thetap = theta[0:-1]
    sigman = theta[-1]

    # first the negloglik
    K, dK = kernel.value_and_grad(thetap, xx)
    assert dK.shape[0] == thetap.size
    Ky = K + np.diag(Sigma) * sigman**2
    u, s, v = np.linalg.svd(Ky, hermitian=True)
    logdetK = np.sum(np.log(s))
    Kyinv = u.dot(v/s.reshape(N, 1))
    alpha = Kyinv.dot(y)
    Z = (np.vdot(y, alpha) + logdetK)/2

    # derivs
    dZ = np.zeros(theta.size)
    alpha = alpha.reshape(N, 1)
    aaT = Kyinv - alpha.dot(alpha.T)

    for k in range(thetap.size):
        dZ[k] = np.sum(diag_dot(aaT, dK[k]))/2

    # deriv wrt sigman
    dK = np.diag(2*sigman*Sigma)
    dZ[-1] = np.sum(diag_dot(aaT, dK))/2

    return Z, dZ


def meanf(xx, xxp, y, Sigma, theta, kernel):
    K = kernel(theta, xx)
    Kp = kernel(theta, xxp)
    Ky = K + np.diag(Sigma * theta[-1]**2)
    Kinvy = np.linalg.solve(Ky, y)
    return Kp @ Kinvy


def meancovf(xx, xxp, xxpp, y, Sigma, theta, kernel):
    N = xx.shape[0]
    K = kernel(theta, xx)
    Kp = kernel(theta, xxp)
    Kpp = kernel(theta, xxpp)
    Ky = K + np.diag(Sigma * theta[-1]**2)
    u, s, v = np.linalg.svd(Ky, hermitian=True)
    Kyinv = u.dot(v/s.reshape(N, 1))
    return Kp.dot(Kyinv.dot(y)), Kpp - Kp.dot(Kyinv.dot(Kp.T))


def gplearn(theta, x, y, w, xp, kernel,
            bounds=((1e-5, None), (1e-5, None), (1e-5, None))):
    """
    Args:
        y (_type_): _description_
        x (_type_): _description_
        w (_type_): _description_
        xp (_type_): _description_
        theta0 (_type_): _description_
        kernel (_type_):
    """
    # drop entries with zero weights
    idxn0 = w!=0
    x = x[idxn0]
    y = y[idxn0]
    w = w[idxn0]

    N = x.size

    # get matrices of differences
    XX = abs_diff(x, x)
    XXp = abs_diff(xp, x)
    XXpp = abs_diff(xp, xp)

    # kgrad = partial(kernel.value_and_grad, xx=XX)
    Sigma = 1.0/w
    theta, fval, dinfo = fmin(dZdtheta,
                              theta,
                              args=(y, Sigma, kernel, XX),
                              approx_grad=False,
                              bounds=bounds,
                              factr=1e6)

    if dinfo['warnflag']:
        print(dinfo['task'])

    muf, covf = meancovf(XX, XXp, XXpp, y, Sigma, theta, kernel)
    return theta, muf, covf


def emterp(theta, x, y, kernel, w=None, xp=None, nu=2, niter=5, tol=1e-3):
    if xp is None:
        xp = x

    if w is None:
        w = np.ones_like(y)

    eps = 1.0
    mup = np.zeros_like(y)
    for i in range(niter):
        theta, muf, covf = gplearn(theta, x, y, w, xp, kernel)

        eps = np.linalg.norm(muf - mup)/np.linalg.norm(muf)

        if eps < tol:
            break

        mup = muf.copy()

        res = y - muf
        ressq = res**2/theta[-1]**2

        # new weights
        w = (nu+1)/(nu + ressq)
        logw = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc,
                        nu,
                        args=(np.mean(w), np.mean(logw)),
                        approx_grad=True,
                        bounds=((1e-1, None),))

    return theta, muf, np.diag(covf)
