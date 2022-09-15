import numpy as np
import numba
from scipy.special import polygamma
from scipy.optimize import fmin_l_bfgs_b as fmin
from smoove.utils import nufunc
from smoove.filter import Kfilter, Kfilter_fast
from smoove.smoother import RTSsmoother, RTSsmoother_fast
from smoove.evidence import evidence, evidence_fast


def kanterp(x, y, w, niter=5, nu0=2, sigmaf0=None, sigman0=1, verbose=0, window=10):
    N = x.size
    M = 2  # cubic smoothing spline
    H = np.zeros((1, M), dtype=np.float64)
    H[0, 0] = 1



    if sigmaf0 is None:
        sigmaf = np.sqrt(N)
    else:
        sigmaf = sigmaf0

    bnds = ((0.1*sigmaf, 10*sigmaf),)
    I = y != 0
    m0 = np.median(y[I][0:window])
    x0 = np.median(x[I][0:window])
    mplus = np.median(y[I][window:2*window])
    xplus = np.median(x[I][window:2*window])
    dm0 = (mplus-m0)/(xplus - x0)
    m0 = np.array((m0, dm0))
    P0 = np.mean((y[I][0:window] - m0)**2) * np.eye(M)
    w /= sigman0**2
    m, P = Kfilter(sigmaf, y, x, w, m0, dm0, P0, 0, P0)
    ms, Ps = RTSsmoother(m, P, x, sigmaf)

    # initial residual
    res = y - ms[0]
    sigman = np.sqrt(np.mean(res**2*w))
    nu = nu0
    for k in range(niter):
        ressq = res**2/sigman**2

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get smoothed signal
        sigmaf, fval, dinfo = fmin(evidence, np.array(sigmaf),
                                   args=(y, x, eta, ms[0, 0], ms[1, 0], Ps[0, 0, 0], Ps[0, 1, 0], Ps[1, 1, 0], sigman),
                                   approx_grad=True,
                                   bounds=bnds)



        m0 = ms[:, 0]
        P0 = Ps[:, :, 0]
        m, P = Kfilter_fast(sigmaf[0], y, x, eta/sigman**2, ms[0, 0], ms[1, 0], Ps[0, 0, 0], Ps[0, 1, 0], Ps[1, 1, 0])
        ms, Ps = RTSsmoother_fast(m, P, x, sigmaf[0])


        if verbose:
            print(f"Z={fval}, sigmaf={sigmaf[0]}, sigman={sigman}, warning={dinfo['warnflag']}")

        if k == niter - 1:
            return ms, Ps

        # residual
        res = y - ms[0]

        sigman = np.sqrt(np.mean(res**2*eta))

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(eta), np.mean(logeta)),
                        approx_grad=True,
                        bounds=((1e-2, None),))


def kanterp_fast(x, y, w, niter=5, nu0=2, sigmaf0=None, sigman0=1, verbose=0, window=10):
    N = x.size
    M = 2  # cubic smoothing spline
    H = np.zeros((1, M), dtype=np.float64)
    H[0, 0] = 1



    if sigmaf0 is None:
        sigmaf = np.sqrt(N)
    else:
        sigmaf = sigmaf0

    bnds = ((0.1*sigmaf, 10*sigmaf),)
    I = y != 0
    m0 = np.median(y[I][0:window])
    x0 = np.median(x[I][0:window])
    mplus = np.median(y[I][window:2*window])
    xplus = np.median(x[I][window:2*window])
    dm0 = (mplus-m0)/(xplus - x0)
    P0 = np.mean((y[I][0:window] - m0)**2)
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    w /= sigman0**2
    m, P = Kfilter_fast(sigmaf, y, x, w, m0, dm0, P0, 0, P0)
    ms, Ps = RTSsmoother_fast(m, P, x, sigmaf)

    # initial residual
    res = y - ms[0]
    sigman = np.sqrt(np.mean(res**2*w))
    nu = nu0
    for k in range(niter):
        ressq = res**2/sigman**2

        # solve for weights
        eta = (nu+1)/(nu + ressq)
        logeta = polygamma(0, (nu+1)/2) - np.log((nu + ressq)/2)

        # get smoothed signal
        sigmaf, fval, dinfo = fmin(evidence3, np.array(sigmaf),
                                   args=(y, x, eta, ms[0, 0], ms[1, 0], Ps[0, 0, 0], Ps[0, 1, 0], Ps[1, 1, 0], sigman),
                                   approx_grad=True,
                                   bounds=bnds)



        m0 = ms[:, 0]
        P0 = Ps[:, :, 0]
        m, P = Kfilter_fast(sigmaf[0], y, x, eta/sigman**2, ms[0, 0], ms[1, 0], Ps[0, 0, 0], Ps[0, 1, 0], Ps[1, 1, 0])
        ms, Ps = RTSsmoother_fast(m, P, x, sigmaf[0])


        if verbose:
            print(f"Z={fval}, sigmaf={sigmaf[0]}, sigman={sigman}, warning={dinfo['warnflag']}")

        if k == niter - 1:
            return ms, Ps

        # residual
        res = y - ms[0]

        sigman = np.sqrt(np.mean(res**2*eta))

        # degrees of freedom nu
        nu, _, _ = fmin(nufunc, nu, args=(np.mean(eta), np.mean(logeta)),
                        approx_grad=True,
                        bounds=((1e-2, None),))
