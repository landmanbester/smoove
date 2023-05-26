import numpy as np
import numba
from numba import njit
from scipy.optimize import fmin_l_bfgs_b as fmin
from scipy.special import polygamma
from scipy import linalg
from time import time


LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype=np.float64)

@numba.njit(cache=True)
def factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]


@numba.njit(cache=True)
def digamma(x):
    '''
    Approximation of the digamma function.
    The eror is O(1e-3) for x = 1 and improves for larger values
    '''
    x2 = x*x
    x4 = x2*x2
    x6 = x4*x2
    return np.log(x) - 1/(2*x) - 1/(12*x2) + 1/(120*x4) - 1/(252*x6)


# @numba.njit(fastmath=True, cache=True)
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


def nufunc(nu, meaneta, meanlogeta):
    const = 1 + meanlogeta - meaneta
    val = polygamma(0, nu/2) - np.log(nu/2) - const
    return val*val


@numba.njit(fastmath=True, inline='always', cache=True)
def Afunc(delta, m):
    # phi = delta**np.arange(m)/list(map(factorial, np.arange(m)))
    phi = np.zeros(m, dtype=np.float64)
    A = np.zeros((m,m), dtype=np.float64)
    for i in range(0, m):
        phi[i] = delta**i/factorial(i)
    A[0, :] = phi
    for i in range(1, m):
        A[i, i:] = phi[0:-i]
    return A


@numba.njit(fastmath=True, inline='always', cache=True)
def Qfunc(q, delta, m):
    Q = np.zeros((m, m), dtype=np.float64)
    for j in range(1, m+1):
        for k in range(1, m+1):
            tmp = ((m-j) + (m-k) + 1)
            Q[j-1, k-1] = q*delta**tmp/(tmp*factorial(m-j)*factorial(m-k))

    return Q


def modulated_chirp(t, a, b, c):
    tm = np.median(t)
    envelope = np.exp(-(t-tm)**2/(2*a**2))
    # chirp = np.sin(b*t**2 + c*t)
    chirp = np.sin(b*np.exp(t**2))
    return envelope * chirp


@numba.njit(fastmath=True, inline='always', cache=True)
def inv(A):
    a = A[0,0]
    b = A[0,1]
    c = A[1,0]
    d = A[1,1]
    det = np.maximum(a*d - b*c, 1e-6)
    return np.array(((d, -b), (-c, a)))/det

# import matplotlib.pyplot as plt
# a = 0.5
# b = 30
# c = 0
# N = 512
# t = np.linspace(-1, 1, N)
# f = modulated_chirp(t, a, b, c)

# plt.plot(t, f)
# plt.show()
