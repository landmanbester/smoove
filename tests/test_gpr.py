from functools import partial
import numpy as np
from smoove.gpr import gplearn, meanf, meancovf
from smoove.kernels.mattern52 import mat52
from smoove.kernels.sqexp import sqexp
import matplotlib.pyplot as plt
import pytest

pmp = pytest.mark.parametrize

def func(x, a, b, c):
    return a*np.sin(b*x)*np.exp(c*x), a*b*np.cos(b*x)*np.exp(c*x) + a*c*np.sin(b*x)*np.exp(c*x)

@pmp("a", (-10, 1, 20))
@pmp("b", (-5, 5))
@pmp("c", (-1, 0, 1))
@pmp("N", (128, 256))
def test_gpr(a, b, c, N):
    np.random.seed(42)
    x = np.sort(np.random.random(N))
    f, df = func(x, a, b, c)
    sigmas = np.ones(N)  #np.exp(np.random.randn(N))
    n = sigmas*np.random.randn(N)
    w = 1/sigmas**2
    y = f + n


    xp = x  #np.linspace(0, 1, 2*N)
    fp, _ = func(xp, a, b, c)
    theta = np.array((np.std(y), 0.25*x.max(), 1.0))
    kernel = mat52()
    # kernel = sqexp()
    theta, muf, covf = gplearn(theta, x, y, w, xp, kernel)

    # plt.fill_between(xp, muf - np.sqrt(np.diag(covf)), muf + np.sqrt(np.diag(covf)))
    # plt.plot(xp, fp, 'k')
    # plt.plot(xp, muf, 'b')
    # plt.errorbar(x, y, sigmas, fmt='xr')
    # plt.show()

    # check that at least 67% of the true function lies in the 1-sigma confidence intervals
    diff = fp - muf
    Iin = np.abs(diff) <= np.sqrt(np.diag(covf))
    frac_in = np.sum(Iin)/Iin.size
    print(frac_in)
    assert frac_in >= 0.5


# @pmp("a", (-10, 1, 20))
# @pmp("b", (-5, 5))
# @pmp("c", (-1, 0, 1))
# @pmp("N", (256, 512))
# def test_emterp(a, b, c, N):
#     x = np.sort(np.random.random(N))
#     f, df = func(x, a, b, c)
#     sigmas = np.exp(np.random.randn(N))
#     n = sigmas*np.random.randn(N)
#     w = 1/sigmas**2
#     y = f + n
#     idx = np.random.randint(0, N, int(0.1*N))
#     y[idx] += 100*np.random.randn(N)








# test_gpr(-10, 5, -1, 128)
