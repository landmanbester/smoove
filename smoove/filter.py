import numpy as np
import numba
from smoove.utils import Afunc, Qfunc

@numba.njit(nogil=True, fastmath=True, cache=True)
def Kfilter(sigmaf, x, y, w, m0, P0, H):
    N = x.size
    delta = x[1:] - x[0:-1]
    M = m0.size
    m = np.zeros((N, M), dtype=np.float64)
    P = np.zeros((N, M, M), dtype=np.float64)
    m[0, :] = m0
    P[0, :, :] = P0
    Z = np.zeros((1,1))  # evidence
    for k in range(1, N):
        A = Afunc(delta[k-1], M)
        Q = Qfunc(sigmaf**2, delta[k-1], M)

        mp = A @ m[k-1, :]
        Pp = A @ P[k-1, :, :] @ A.T + Q

        if w[k] > 1e-6:
            v = y[k] - H @ mp

            # Use WMI to write inverse ito weights (not variance)
            Ppinv = np.linalg.inv(Pp)
            tmp = Ppinv + H.T @ (w[k] * H)
            tmpinv = np.linalg.inv(tmp)
            Sinv = w[k] - w[k] * H @ tmpinv @ (H.T * w[k])

            Z += v*Sinv*v - np.log(Sinv)

            K = Pp @ H.T @ Sinv

            m[k, :] = mp + K @ v
            P[k, :, :] = Pp - K @ H @ Pp
        else:
            m[k, :] = mp
            P[k, :, :] = Pp

    return m, P, Z


@numba.njit(nogil=True, fastmath=True, cache=True)
def Kfilter2(sigmaf, x, y, w, m0, P0, H):
    N = x.size
    delta = x[1:] - x[0:-1]
    M = m0.size
    m = np.zeros((N, M), dtype=np.float64)
    P = np.zeros((N, M, M), dtype=np.float64)
    m[0, :] = m0
    P[0, :, :] = P0
    Z = np.zeros((1,1))  # evidence
    R = 1.0/w
    for k in range(1, N):
        A = Afunc(delta[k-1], M)
        Q = Qfunc(sigmaf**2, delta[k-1], M)

        mp = A @ m[k-1, :]
        Pp = A @ P[k-1, :, :] @ A.T + Q


        v = y[k] - H @ mp
        S = H @ Pp @ H.T + R[k]
        Sinv = np.linalg.inv(S)

        Z += 0.5*v*Sinv*v + 0.5*np.log(2*np.pi * S)

        K = Pp @ H.T @ Sinv

        m[k, :] = mp + K @ v
        P[k, :, :] = Pp - K @ H @ Pp
        # P[k, :, :] = Pp - K @ S @ K.T


    return m, P, Z
