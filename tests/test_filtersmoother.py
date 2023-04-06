import numpy as np
from smoove.filter import Kfilter, Kfilter2
from smoove.smoother import RTSsmoother
import matplotlib.pyplot as plt
import pytest

pmp = pytest.mark.parametrize

def func(x, a, b, c):
    return a*np.sin(b*x)*np.exp(c*x), a*b*np.cos(b*x)*np.exp(c*x) + a*c*np.sin(b*x)*np.exp(c*x)

@pmp("a", (-10, 1, 20))
@pmp("b", (-5, 5))
@pmp("c", (-1, 0, 1))
@pmp("N", (128, 512))
def test_Kfilter(a, b, c, N):
    x = np.linspace(0.1, 0.9, N)   #np.sort(np.random.random(N))
    f, df = func(x, a, b, c)
    sigmas = np.ones(N)  #np.exp(np.random.randn(N))
    n = sigmas*np.random.randn(N)
    w = 1/sigmas**2
    y = f + n

    # find optimal sigmaf by brute force
    sigmafs = np.linspace(0.1, 2*N, 2*N)
    sigman = 1.0
    m0 = np.array((f[0], df[0]), dtype=np.float64)
    P0 = np.array(((np.var(n), 0), (0, 1)), dtype=np.float64)
    H = np.zeros((1, 2), dtype=np.float64)
    H[0, 0] = 1
    Zs = np.zeros(2*N)
    for i, sigmaf in enumerate(sigmafs):
        muf, covf, Z = Kfilter2(sigmaf, y, x, w, m0, P0, H)
        Zs[i] = Z.squeeze()

    idx = np.where(Zs == Zs.min())[0][0]
    sigmaf = sigmafs[idx]
    muf, covf, Z = Kfilter2(sigmaf, y, x, w, m0, P0, H)
    mus, covs = RTSsmoother(muf, covf, x, sigmaf)

    # plt.fill_between(x, mus[:, 0] - np.sqrt(covs[:, 0, 0]), mus[:, 0] + np.sqrt(covs[:, 0, 0]))
    # plt.plot(x, f, 'k')
    # plt.plot(x, mus[:, 0], 'b')
    # plt.errorbar(x, y, sigmas, fmt='xr')
    # plt.show()

    # check that at least 67% of the true function lies in the 1-sigma confidence intervals
    diff = f - mus[:, 0]
    Iin = np.abs(diff) <= np.sqrt(covs[:, 0, 0])
    frac_in = np.sum(Iin)/Iin.size
    assert frac_in >= 0.5





test_Kfilter(-10, 5, -1, 512)
