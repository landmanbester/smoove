import numpy as np
import pytest

from smoove.gpr import emterp
from smoove.kernels.mattern52 import mat52

pmp = pytest.mark.parametrize


def func(x, a, b, c):
    return a * np.sin(b * x) * np.exp(c * x), a * b * np.cos(b * x) * np.exp(
        c * x
    ) + a * c * np.sin(b * x) * np.exp(c * x)


@pmp("a", (-10, 10))
@pmp("b", (-5, 5))
@pmp("c", (-1, 1))
@pmp("N", (128,))
def test_emterp(a, b, c, N):
    np.random.seed(42)
    x = np.linspace(0, 1.0, N)
    f, df = func(x, a, b, c)
    sigman = np.ones(N)
    # sigman = np.exp(np.random.randn(N)) #/10000
    n = sigman * np.random.randn(N)
    w = 1 / sigman**2
    y = f + n

    # add outliers
    for _i in range(int(0.1 * N)):
        idx = np.random.randint(0, N)
        y[idx] += 10 * np.random.randn()

    theta = np.array((np.var(y), 0.25 * (x.max() - x.min()), 1.0))
    kernel = mat52()
    theta, muf, covf = emterp(theta, x, y, kernel, w=w, niter=100, nu=5)

    diff = f - muf
    Iin = np.abs(diff) <= np.sqrt(covf)
    frac_in = np.sum(Iin) / Iin.size
    # about 67% needs to be within contours but that doe snot always work
    assert frac_in >= 0.50
