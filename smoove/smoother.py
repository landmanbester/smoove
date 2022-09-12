import numpy as np
from smoove.utils import Afunc, Qfunc

# @numba.njit
def RTSsmoother(m, P, x, sigmaf):
    K = x.size
    delta = np.zeros(K)
    delta[0] = 100
    delta = x[1:] - x[0:-1]
    M = m[:, 0].size
    ms = np.zeros((M, K), dtype=np.float64)
    Ps = np.zeros((M, M, K), dtype=np.float64)
    ms[:, K-1] = m[:, K-1]
    Ps[:, :, K-1] = P[:, :, K-1]

    for k in range(K-2, -1, -1):
        A = Afunc(delta[k], M)
        Q = Qfunc(sigmaf**2, delta[k], M)

        mp = A @ m[:, k]
        Pp = A @ P[:, :, k] @ A.T + Q
        Pinv = np.linalg.inv(Pp)

        G = P[:, :, k] @ A.T @ Pinv

        ms[:, k] = m[:, k] + G @ (ms[:, k+1] - mp)
        Ps[:, :, k] = P[:, :, k] + G @ (Ps[:, :, k+1] - Pp) @ G.T

    return ms, Ps


@numba.njit
def RTSsmoother_fast(m, P, x, sigmaf):
    K = x.size
    delta = np.zeros(K)
    delta = x[1:] - x[0:-1]
    ms = np.zeros((2, K), dtype=np.float64)
    Ps = np.zeros((2, 2, K), dtype=np.float64)
    ms[:, K-1] = m[:, K-1]
    Ps[:, :, K-1] = P[:, :, K-1]

    a00 = a11 = 1
    q = sigmaf**2
    for k in range(K-2, -1, -1):
        dlta = delta[k]

        a01 = dlta
        qd = q*dlta
        qdd = qd * dlta
        q00 = qdd * dlta / 3
        q01 = qdd/2
        q11 = qd

        mp0 = m[0, k] + dlta * m[1, k]
        mp1 = m[1, k]
        pp00 = dlta*P[0, 1, k] + dlta*(dlta*P[1, 1, k] + P[0, 1, k]) + P[0, 0, k] + q00
        pp01 = dlta*P[1, 1, k] + P[0, 1, k] + q01
        pp11 = P[1, 1, k] + q11

        # import pdb; pdb.set_trace()

        det = pp00*pp11 - pp01*pp01

        g00 = (-P[0,1,k]*pp01 + pp11*(dlta*P[0, 1, k] + P[0, 0, k]))/det
        g01 = (P[0,1,k]*pp00 - pp01*(dlta*P[0,1,k] + P[0,0,k]))/det
        g10 = (-P[1,1,k]*pp01 + pp11*(dlta*P[1,1,k] + P[0,1,k]))/det
        g11 = (P[1,1,k]*pp00 - pp01*(dlta*P[1,1,k] + P[0,1,k]))/det

        ms[0, k] = m[0, k] + g00 * (ms[0, k+1] - mp0) + g01 * (ms[1, k+1] - mp1)
        ms[1, k] = m[1, k] + g10 * (ms[0, k+1] - mp0) + g11 * (ms[1, k+1] - mp1)

        Ps[0, 0, k] = P[0, 0, k] - g00*(g00*(pp00 - Ps[0, 0, k+1]) + g01*(pp01 - Ps[0, 1, k+1])) - g01*(g00*(pp01 - Ps[0, 1, k+1]) + g01*(pp11 - Ps[1, 1, k+1]))
        Ps[0, 1, k] = P[0, 1, k] - g10*(g00*(pp00 - Ps[0, 0, k+1]) + g01*(pp01 - Ps[0, 1, k+1])) - g11*(g00*(pp01 - Ps[0, 1, k+1]) + g01*(pp11 - Ps[1, 1, k+1]))
        Ps[1, 0, k] = P[1, 0, k] - g00*(g10*(pp00 - Ps[0, 0, k+1]) + g11*(pp01 - Ps[0, 1, k+1])) - g01*(g10*(pp01 - Ps[0, 1, k+1]) + g11*(pp11 - Ps[1, 1, k+1]))
        Ps[1, 1, k] = P[1, 1, k] - g10*(g10*(pp00 - Ps[0, 0, k+1]) + g11*(pp01 - Ps[0, 1, k+1])) - g11*(g10*(pp01 - Ps[0, 1, k+1]) + g11*(pp11 - Ps[1, 1, k+1]))

    return ms, Ps
