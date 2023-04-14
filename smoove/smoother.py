import numpy as np
import numba
from smoove.utils import Afunc, Qfunc

@numba.njit(nogil=True, fastmath=True, cache=True)
def RTSsmoother(m, P, x, sigmaf):
    N = x.size
    delta = x[1:] - x[0:-1]
    M = m[0, :].size
    ms = np.zeros((N, M), dtype=np.float64)
    Ps = np.zeros((N, M, M), dtype=np.float64)
    ms[N-1, :] = m[N-1, :]
    Ps[N-1, :, :] = P[N-1, :, :]

    for k in range(N-2, -1, -1):
        A = Afunc(delta[k], M)
        Q = Qfunc(sigmaf**2, delta[k], M)

        mp = A @ m[k, :]
        Pp = A @ P[k, :, :] @ A.T + Q
        Pinv = np.linalg.inv(Pp)

        G = P[k, :, :] @ A.T @ Pinv

        ms[k, :] = m[k, :] + G @ (ms[k+1, :] - mp)
        Ps[k, :, :] = P[k, :, :] + G @ (Ps[k+1, :, :] - Pp) @ G.T

    return ms, Ps
