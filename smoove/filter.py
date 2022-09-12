import numpy as np
from smoove.utils import Afunc, Qfunc

# @numba.njit
def Kfilter(m0, P0, x, y, H, Rinv, sigmaf):
    N = x.size
    delta = x[1:] - x[0:-1]
    M = m0.size
    m = np.zeros((M, N), dtype=np.float64)
    P = np.zeros((M, M, N), dtype=np.float64)
    m[:, 0] = m0
    P[:, :, 0] = P0

    for k in range(1, N):
        A = Afunc(delta[k-1], M)
        Q = Qfunc(sigmaf**2, delta[k-1], M)

        mp = A @ m[:, k-1]
        Pp = A @ P[:, :, k-1] @ A.T + Q

        if Rinv[k]:
            v = y[k] - H @ mp

            # Use WMI to write inverse ito weights (not variance)
            Ppinv = np.linalg.inv(Pp)
            tmp = Ppinv + H.T @ (Rinv[k] * H)
            tmpinv = np.linalg.inv(tmp)
            Sinv = Rinv[k] - Rinv[k] * H @ tmpinv @ (H.T * Rinv[k])

            K = Pp @ H.T @ Sinv

            m[:, k] = mp + K @ v
            P[:, :, k] = Pp - K @ H @ Pp
        else:
            m[:, k] = mp
            P[:, :, k] = Pp

    return m, P


@numba.njit
def Kfilter_fast(sigmaf, y, x, Rinv, m0, m1, p00, p01, p11):
    N = x.size
    delta = x[1:] - x[0:-1]
    m = np.zeros((2, N), dtype=np.float64)
    P = np.zeros((2, 2, N), dtype=np.float64)
    m[0, 0] = m0
    m[1, 0] = m1
    P[0, 0, 0] = p00
    P[0, 1, 0] = p01
    P[1, 0, 0] = p01
    P[1, 1, 0] = p11

    q = sigmaf**2
    w = Rinv
    for k in range(1, N):
        # This can be avoided if the data are on a regular grid
        dlta = delta[k-1]
        a01 = dlta
        qd = q*dlta
        qdd = qd * dlta
        q00 = qdd * dlta / 3
        q01 = qdd/2
        q11 = qd

        mp0 = m[0, k-1] + dlta * m[1, k-1]
        mp1 = m[1, k-1]
        pp00 = dlta*P[0, 1, k-1] + dlta*(dlta*P[1, 1, k-1] + P[0, 1, k-1]) + P[0, 0, k-1] + q00
        pp01 = dlta*P[1, 1, k-1] + P[0, 1, k-1] + q01
        pp11 = P[1, 1, k-1] + q11

        if Rinv[k]:
            v = y[k] - mp0
            det = pp00 * pp11 - pp01 * pp01

            a2 = pp11/det + w[k]
            b2 = -pp01/det
            c2 = -pp01/det
            d2 = pp00/det
            det2 = a2*d2 - b2 * c2
            Sinv = w[k] - w[k]**2 * d2 / det2

            m0 = mp0 + pp00 * Sinv * v
            m1 = mp1 + pp01 * Sinv * v
            p00 = -pp00**2*Sinv + pp00
            p01 = -pp00*pp01*Sinv + pp01
            p11 = -pp01*Sinv*pp01 + pp11

            m[0, k] = mp0 + pp00 * Sinv * v
            m[1, k] = mp1 + pp01 * Sinv * v
            P[0, 0, k] = -pp00**2*Sinv + pp00
            P[0, 1, k] = -pp00*pp01*Sinv + pp01
            P[1, 0, k] = P[0, 1, k]
            P[1, 1, k] = -pp01*Sinv*pp01 + pp11
        else:
            m[0, k] = mp0
            m[1, k] = mp1
            P[0, 0, k] = pp00
            P[0, 1, k] = pp01
            P[1, 0, k] = pp01
            P[1, 1, k] = pp11


    return m, P
