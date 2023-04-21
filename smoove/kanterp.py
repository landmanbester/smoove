import numpy as np
import numba
from scipy.special import polygamma
from scipy.optimize import fmin_l_bfgs_b as fmin
from smoove.utils import nufunc
from smoove.filter import Kfilter
from smoove.smoother import RTSsmoother
from time import time


def evidence(theta, x, y, w, m0, P0, H):
    sigmaf  = theta[0]
    sigman = theta[-1]
    muf, covf, Z = Kfilter(sigmaf, x, y, w/sigman**2, m0, P0, H)
    return Z


def kanterp(x, y, w, niter=5, nu=2, tol=1e-3):
    '''

    General algorithm for smoothing with a Student's t-distribution.
    We assume data in the form

    y = f(x) + epsilon

    where epsilon can be drawn from a Student's t-distribution. The aim
    is to reconstruct the function f(.).

    Inputs:

        x       - coordinates
        y       - data
        w       - initial weights
        niter   - number of EM iterations
        nu0     - initial guess for the degrees of freedom parameter

    '''
    ti = time()
    # check x is in (0,1)
    assert x.min() > 0.0
    assert x.max() < 1.0
    # check x is sorted
    assert (x[1:] - x[0:-1] > 0.0).all()


    N = x.size
    M = 2  # 2nd order IWP
    H = np.zeros((1, M), dtype=np.float64)
    H[0, 0] = 1

    theta = np.array((np.sqrt(N), 1.0))

    bnds = ((1e-2, 10*N), (1e-2, 1e5))
    I = w != 0.0
    med0 = np.median(y[I][0:5])
    m0 = np.array((med0, 1.0))
    # P0 = np.array(((np.var(y[I][0:5] - med0), 0.0), (0.0, 1.0)))
    P0 = np.eye(M)

    mup = np.zeros_like(y)
    for k in range(niter):
        theta, fval, dinfo = fmin(evidence, theta,
                                  args=(x, y, w, m0, P0, H),
                                  approx_grad=True,
                                  bounds=bnds) #,
                                #   epsilon=1e-3)
        sigmaf = theta[0]
        sigman = theta[-1]

        m, P, Z = Kfilter(sigmaf, x, y, w/sigman**2, m0, P0, H)
        ms, Ps = RTSsmoother(m, P, x, sigmaf)
        muf = ms[:, 0]

        if np.isnan(muf).any():
            print(k, np.sqrt(N), sigmaf, sigman)
            print(dinfo)
            m0 = np.array((med0, 1.0))
            P0 = np.eye(M)
            mup = np.zeros_like(y)
            sigmaf /= 2
            sigman *= 10
            theta = np.array((sigmaf, sigman))
            continue
        else:
            m0 = ms[0]
            P0 = 2*np.diag(np.diag(Ps[0]))  # LB - why the 2?

            eps = np.linalg.norm(muf-mup)/np.linalg.norm(muf)
            if eps < tol:
                break

            mup = muf.copy()

        res = y - muf
        ressq = res**2/sigman**2

        # solve for weights
        w = (nu+1)/(nu + ressq)
        logw = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(w), np.mean(logw)),
                        approx_grad=True,
                        bounds=((1e-1, None),))

    # print(time() - ti)
    return theta, muf, Ps[:, 0, 0]
