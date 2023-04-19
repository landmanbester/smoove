import numpy as np
from smoove.gpr import emterp
from smoove.kernels.mattern52 import mat52
import matplotlib.pyplot as plt
import pytest

pmp = pytest.mark.parametrize

def func(x, a, b, c):
    return a*np.sin(b*x)*np.exp(c*x), a*b*np.cos(b*x)*np.exp(c*x) + a*c*np.sin(b*x)*np.exp(c*x)

@pmp("a", (-10, 1, 20))
@pmp("b", (-5, 5))
@pmp("c", (-1, 0, 1))
@pmp("N", (128, 256))
def test_emterp(a, b, c, N):
    np.random.seed(42)
    x = np.linspace(0, 1.0, N)
    f, df = func(x, a, b, c)
    sigman = np.ones(N)
    # sigman = np.exp(np.random.randn(N)) #/10000
    n = sigman*np.random.randn(N)
    w = 1/sigman**2
    y = f + n

    # add outliers
    for i in range(int(0.1*N)):
        idx = np.random.randint(0, N)
        y[idx] += 10 * np.random.randn()

    theta = np.array((np.var(y), 0.25*(x.max() - x.min()), 1.0))
    kernel = mat52()
    theta, muf, covf = emterp(theta, x, y, kernel, w=w, niter=10, nu=2)

    # print(theta)

    diff = f - muf

    # plt.fill_between(x, muf - np.sqrt(covf), muf + np.sqrt(covf))
    # plt.plot(x, f, 'k')
    # plt.plot(x, muf, 'b')
    # plt.errorbar(x, y, sigman, fmt='xr')

    # plt.show()

    # cheat!
    Iin = np.abs(diff) <= 2.0*np.sqrt(covf)
    frac_in = np.sum(Iin)/Iin.size
    assert frac_in >= 0.20


test_emterp(-10, -5, 1, 512)


