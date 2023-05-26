
import numpy as np
import numba
from smoove.utils import Afunc, Qfunc, inv
# inv = np.linalg.inv
# import warnings
# warnings.filterwarnings("error")

@numba.njit(nogil=True, cache=True)
def Kfilter(sigmaf, x, y, w, m0, P0, H):
    N = x.size
    delta = x[1:] - x[0:-1]
    M = m0.size
    m = np.zeros((N, M), dtype=np.float64)
    P = np.zeros((N, M, M), dtype=np.float64)
    m[0, :] = m0
    P[0, :, :] = P0
    Z = np.zeros((1,1))  # evidence
    # Z2 = 0.0
    for k in range(1, N):
        A = Afunc(delta[k-1], M)
        AT = (A.T).copy()
        Q = Qfunc(sigmaf**2, delta[k-1], M)

        mp = A @ m[k-1]
        Pp = A @ P[k-1] @ AT + Q

        if w[k] > 1e-6:
            v = y[k] - H @ mp
            # v2 = y[k] - mp[0]
            # print(v, v2)

            # Use WMI to write inverse ito weights (not variance)
            Ppinv = inv(Pp)
            # Ppinv[0, 0] += w[k]
            tmp = Ppinv + H.T @ (w[k] * H)
            tmpinv = inv(tmp)
            # tmpinv = inv(Ppinv)

            # tmpinv = inv(Ppinv)
            Sinv = w[k] - w[k] * H @ tmpinv @ (H.T * w[k])
            # Sinv2 = w[k] - w[k] **2 * tmpinv[0,0]

            Z += v*Sinv*v - np.log(Sinv)

            # Z2 += v2*Sinv2*v2 - np.log(Sinv2)

            K = Pp @ (H.T * Sinv)
            # K2 = Pp[:, 0] * Sinv2



            m[k] = mp + K @ v
            # m[k, :] = mp + K2 * v2
            # print(m[k,:] - (mp + K2 * v2))
            P[k] = Pp - K @ H @ Pp

        else:
            m[k] = mp
            P[k] = Pp

    return m, P, Z


@numba.njit(nogil=True, fastmath=True, cache=True)
def Kfilter2(sigmaf, x, y, w, m0, P0, H):
    N = x.size
    delta = x[1:] - x[0:-1]
    M = m0.size
    m = np.zeros((N, M), dtype=np.float64)
    P = np.zeros((N, M, M), dtype=np.float64)
    m[0] = m0
    P[0] = P0
    Z = np.zeros((1,1))  # evidence
    R = 1.0/w
    for k in range(1, N):
        A = Afunc(delta[k-1], M)
        Q = Qfunc(sigmaf**2, delta[k-1], M)

        mp = A @ m[k-1]
        Pp = A @ P[k-1] @ A.T + Q


        v = y[k] - H @ mp
        S = H @ Pp @ H.T + R[k]
        Sinv = np.linalg.inv(S)

        Z += 0.5*v*Sinv*v + 0.5*np.log(2*np.pi * S)

        K = Pp @ H.T @ Sinv

        m[k] = mp + K @ v
        P[k] = Pp - K @ H @ Pp
        # P[k, :, :] = Pp - K @ S @ K.T


    return m, P, Z
