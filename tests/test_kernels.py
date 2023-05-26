import numpy as np
from smoove.kernels.sqexp import sqexp
from smoove.utils import abs_diff

def test_sqexp():
    x = np.linspace(0, 1, 100)
    XX = abs_diff(x, x)

    k = sqexp()

    theta = np.array((1.414, 0.1))

    def expsq(theta, xx):
        return theta[0]**2 * np.exp(-xx**2/(2*theta[1]**2))

    K1 = expsq(theta, XX)

    K, dK = k.value_and_grad(theta, XX)

    assert np.allclose(K1, K)

    # analytic
    dKs = 2 * K1 / theta[0]
    dKl = XX**2 * K1 / theta[1] ** 3

    assert np.allclose(dKs, dK[0])
    assert np.allclose(dKl, dK[1])
