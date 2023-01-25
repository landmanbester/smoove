import numpy as np
import numba
from smoove.utils import Afunc, Qfunc


def evidence(theta, y, x, H, w, m0, P0, sigman):
    sigmaf = theta[0]
    N = x.size
    delta = x[1:] - x[0:-1]
    M = 2  # cubic spline
    m = np.zeros((M, N), dtype=np.float64)
    P = np.zeros((M, M, N), dtype=np.float64)
    m[:, 0] = m0
    P[:, :, 0] = P0

    Z = 0
    w /= sigman**2
    wmax = w.max()
    for k in range(1, N):
        A = Afunc(delta[k-1], M)
        Q = Qfunc(sigmaf**2, delta[k-1], M)

        mp = A @ m[:, k-1]
        Pp = A @ P[:, :, k-1] @ A.T + Q

        if w[k]/wmax > 1e-6:

            v = y[k] - H @ mp


            # a = Pp[0, 0]
            # # b = Pp[0, 1]
            # # c = Pp[1, 0]
            # b = c = (Pp[0, 1] + Pp[1, 0])/2
            # d = Pp[1, 1]
            # det = a*d - b*c
            # Ppinv = np.array(((d, -b), (-c, a)), dtype=np.float64)/det

            # Use WMI to write inverse ito weights (not variance)
            Ppinv = np.linalg.inv(Pp)

            # import pdb; pdb.set_trace()
            tmp = Ppinv + H.T @ (w[k] * H)
            tmpinv = np.linalg.inv(tmp)

            # a2 = tmp[0, 0]
            # b2 = tmp[0, 1]
            # c2 = tmp[1, 0]
            # d2 = tmp[1, 1]

            # det = a2*d2 - b2*c2

            # tmpinv = np.array(((d2, -b2), (-c2, a2)), dtype=np.float64)/det

            # import pdb; pdb.set_trace()

            Sinv = w[k] - w[k] * H @ tmpinv @ (H.T * w[k])

            Z += 0.5*np.log(2*np.pi/Sinv) + 0.5*v*v*Sinv

            K = Pp @ H.T @ Sinv
            m[:, k] = mp + K @ v
            P[:, :, k] = Pp - K @ H @ Pp
        else:
            m[:, k] = mp
            P[:, :, k] = Pp

    return Z


@numba.njit
def evidence_fast(theta, y, x, w, m0, m1, p00, p01, p11, sigman):
    '''
    Same as evidence fast but theta is just sigmaf
    '''
    N = x.size
    delta = x[1:] - x[0:-1]

    sigmaf = theta[0]

    Z = 0
    w /= sigman**2
    q = sigmaf**2
    a00 = a11 = 1.0
    wmax = w.max()
    for k in range(1, N):
        # This can be avoided if the data are on a regular grid
        dlta = delta[k-1]
        a01 = dlta
        qd = q*dlta
        qdd = qd * dlta
        q00 = qdd * dlta / 3
        q01 = qdd/2
        q11 = qd

        mp0 = m0 + dlta * m1
        mp1 = m1
        pp00 = dlta*p01 + dlta*(dlta*p11 + p01) + p00 + q00
        pp01 = dlta*p11 + p01 + q01
        pp11 = p11 + q11

        if w[k]/wmax > 1e-6:
            v = y[k] - mp0
            det = pp00 * pp11 - pp01 * pp01

            a2 = pp11/det + w[k]
            b2 = -pp01/det
            c2 = -pp01/det
            d2 = pp00/det
            det2 = a2*d2 - b2 * c2
            Sinv = w[k] - w[k]**2 * d2 / det2

            Z += 0.5*np.log(2*np.pi/Sinv) + 0.5*v*v*Sinv

            m0 = mp0 + pp00 * Sinv * v
            m1 = mp1 + pp01 * Sinv * v
            p00 = -pp00**2*Sinv + pp00
            p01 = -pp00*pp01*Sinv + pp01
            p11 = -pp01*Sinv*pp01 + pp11

        else:
            m0 = mp0
            m1 = mp1
            p00 = pp00
            p01 = pp01
            p11 = pp11

    return Z
