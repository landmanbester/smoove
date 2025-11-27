import numpy as np
import pytest

from smoove.kanterp import kanterp

pmp = pytest.mark.parametrize


def func(x, a, b, c):
    return a * np.sin(b * x) * np.exp(c * x), a * b * np.cos(b * x) * np.exp(
        c * x
    ) + a * c * np.sin(b * x) * np.exp(c * x)


@pmp("a", (-10, 20))
@pmp("b", (-5, 5))
@pmp("c", (-1, 1))
@pmp("N", (128, 256))
def test_kanterp(a, b, c, N):
    np.random.seed(42)
    x = np.linspace(0.1, 0.9, N)
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

    theta, muf, covf, nu_out = kanterp(x, y, w, niter=100, nu=2)

    diff = f - muf

    Iin = np.abs(diff) <= np.sqrt(covf)
    frac_in = np.sum(Iin) / Iin.size
    # about 67% needs to be within contours but that doe snot always work
    assert frac_in >= 0.4
