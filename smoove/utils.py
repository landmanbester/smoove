import numpy as np
import numba
from numba import njit
from math import factorial
from scipy.optimize import fmin_l_bfgs_b as fmin
from scipy.special import polygamma
from scipy import linalg
from time import time

def mattern52(xx, sigmaf, l):
    return sigmaf**2*np.exp(-np.sqrt(5)*xx/l)*(1 + np.sqrt(5)*xx/l +
                                               5*xx**2/(3*l**2))


def abs_diff(x, xp):
    try:
        N, D = x.shape
        Np, D = xp.shape
    except Exception:
        N = x.size
        D = 1
        Np = xp.size
        x = np.reshape(x, (N, D))
        xp = np.reshape(xp, (Np, D))
    xD = np.zeros([D, N, Np])
    xpD = np.zeros([D, N, Np])
    for i in range(D):
        xD[i] = np.tile(x[:, i], (Np, 1)).T
        xpD[i] = np.tile(xp[:, i], (N, 1))
    return np.linalg.norm(xD - xpD, axis=0)


@njit(nogil=True, cache=True, inline='always')
def diag_dot(A, B):
    N = A.shape[0]
    C = np.zeros(N)
    for i in range(N):
        for j in range(N):
            C[i] += A[i, j] * B[j, i]
    return C


# @jit(nopython=True, nogil=True, cache=True)
def dZdtheta(theta, xx, y, Sigma):
    '''
    Return log-marginal likelihood and derivs
    '''
    N = xx.shape[0]
    sigmaf = theta[0]
    l = theta[1]
    sigman = theta[2]

    # first the negloglik
    K = mattern52(xx, sigmaf, l)
    Ky = K + np.diag(Sigma) * sigman**2
    # with numba.objmode: # args?
    u, s, v = np.linalg.svd(Ky, hermitian=True)
    logdetK = np.sum(np.log(s))
    Kyinv = u.dot(v/s.reshape(N, 1))
    alpha = Kyinv.dot(y)
    Z = (np.vdot(y, alpha) + logdetK)/2

    # # derivs
    # dZ = np.zeros(theta.size)
    # alpha = alpha.reshape(N, 1)
    # aaT = Kyinv - alpha.dot(alpha.T)

    # # deriv wrt sigmaf
    # dK = 2 * K / sigmaf
    # dZ[0] = np.sum(diag_dot(aaT, dK))/2

    # # deriv wrt l
    # dK = xx * K / l ** 3
    # dZ[1] = np.sum(diag_dot(aaT, dK))/2

    # # deriv wrt sigman
    # dK = np.diag(2*sigman*Sigma)
    # dZ[2] = np.sum(diag_dot(aaT, dK))/2

    return Z  #, dZ

def meanf(xx, xxp, y, Sigma, theta):
    K = mattern52(xx, theta[0], theta[1])
    Kp = mattern52(xxp, theta[0], theta[1])
    Ky = K + np.diag(Sigma) * theta[2]**2
    Kinvy = np.linalg.solve(Ky, y)
    return Kp @ Kinvy

def meancovf(xx, xxp, xxpp, y, Sigma, theta):
    N = xx.shape[0]
    K = mattern52(xx, theta[0], theta[1])
    Kp = mattern52(xxp, theta[0], theta[1])
    Kpp = mattern52(xxpp, theta[0], theta[1])
    Ky = K + np.diag(Sigma) * theta[2]**2
    u, s, v = np.linalg.svd(Ky, hermitian=True)
    Kyinv = u.dot(v/s.reshape(N, 1))
    return Kp.dot(Kyinv.dot(y)), np.diag(Kpp - Kp.T.dot(Kyinv.dot(Kp)))

def gpr(y, x, w, xp, theta0=None, nu=3.0, niter=5):
    # drop entries with zero weights
    idxn0 = w!=0
    x = x[idxn0]
    y = y[idxn0]
    w = w[idxn0]

    N = x.size

    # get matrix of differences
    XX = abs_diff(x, x)
    XXp = abs_diff(x, xp)
    XXpp = abs_diff(xp, xp)

    if theta0 is None:
        theta0 = np.zeros(3)
        theta0[0] = np.std(y)
        theta0[1] = 0.5
        theta0[2] = 1

    # get initial hypers assuming weight scaling
    w = np.ones(N)/np.var(y)
    theta, fval, dinfo = fmin(dZdtheta, theta0, args=(XX, y, 1/w), approx_grad=True,
                              bounds=((1e-5, None), (1e-3, None), (1e-5, 100)))

    mu = meanf(XX, XX, y, 1/w, theta)
    # print(theta)
    # return meancovf(XX, XXp, XXpp, y, 1/w, theta)

    res = y - mu
    for k in range(niter):
        ressq = res**2/theta[-1]**2

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get initial hypers assuming weight scaling
        theta, fval, dinfo = fmin(dZdtheta, theta, args=(XX, y, 1/eta), approx_grad=True,
                                bounds=((1e-5, None), (1e-3, None), (1e-3, 100)))

        if k == niter - 1:
            print(nu, theta)
            return meancovf(XX, XXp, XXpp, y, 1/eta, theta)
        else:
            mu = meanf(XX, XX, y, 1/eta, theta)

        res = y - mu

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(eta), np.mean(logeta)),
                        approx_grad=True,
                        bounds=((1e-2, None),))


def nufunc(nu, meaneta, meanlogeta):
    const = 1 + meanlogeta - meaneta
    val = polygamma(0, nu/2) - np.log(nu/2) - const
    return val*val


# @numba.njit(fastmath=True, inline='always')
def Afunc(delta, m):
    phi = delta**np.arange(m)/list(map(factorial, np.arange(m)))
    A = np.zeros((m,m), dtype=np.float64)
    A[0, :] = phi
    for i in range(1, m):
        A[i, i:] = phi[0:-i]
    return A


# @numba.njit(fastmath=True, inline='always')
def Qfunc(q, delta, m):
    Q = np.zeros((m, m), dtype=np.float64)
    for j in range(1, m+1):
        for k in range(1, m+1):
            tmp = ((m-j) + (m-k) + 1)
            Q[j-1, k-1] = q*delta**tmp/(tmp*factorial(m-j)*factorial(m-k))

    return Q






























