import numpy as np
np.random.seed(420)
from numpy.testing import assert_allclose
from smoove.evidence import evidence, evidence_fast
import pytest

pmp = pytest.mark.parametrize

def func(x, a, b, c):
    return a*np.sin(b*x)*np.exp(c*x), a*b*np.cos(b*x)*np.exp(c*x) + a*c*np.sin(b*x)*np.exp(c*x)

@pmp("a", (-10, 1, 20))
@pmp("b", (-5, 5, 30))
@pmp("c", (-1, 0, 1))
@pmp("N", (128, 512))
def test_evidence(a, b, c, N):
    x = np.sort(np.random.random(N))
    f, df = func(x, a, b, c)
    sigmas = np.exp(np.random.randn(N))
    n = sigmas*np.random.randn(N)
    w = 1/sigmas**2
    y = f + n

    Nsigma = 100
    sigmafs = np.linspace(0.1, N, Nsigma)
    Z1 = np.zeros(Nsigma)
    Z2 = np.zeros(Nsigma)
    sigman = 1.0
    m0 = np.array((f[0], df[0]), dtype=np.float64)
    P0 = np.array(((np.var(n), 0), (0, 1)), dtype=np.float64)
    H = np.zeros((1, 2), dtype=np.float64)
    H[0, 0] = 1
    for i, sigmaf in enumerate(sigmafs):
        theta = np.array([sigmaf])
        Z1[i] = evidence(theta, y, x, H, w, m0, P0, sigman)
        Z2[i] = evidence_fast(theta, y, x, w, m0[0], m0[1],
                              P0[0, 0], P0[0, 1], P0[1, 1], sigman)

    # the evidence can be quite large hence the large atol
    assert_allclose(Z1, Z2, rtol=1e-13, atol=1e-7)

