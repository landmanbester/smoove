import sympy as sm
import numpy as np


class mat52(object):
    def __init__(self):
        sigmaf, l, xx = sm.symbols('sigma l xx', real=True)
        kernel = sigmaf**2 * sm.exp(-sm.sqrt(5) * xx/l) * (1 + sm.sqrt(5)*xx/l + 5*xx**2/(3*l**2))
        dkdsig = kernel.diff(sigmaf)
        dkdl = kernel.diff(l)

        # covariance function and derivs
        self.f = sm.lambdify((sigmaf, l, xx), kernel)
        self.dfds = sm.lambdify((sigmaf, l, xx), dkdsig)
        self.dfdl = sm.lambdify((sigmaf, l, xx), dkdl)

        # state space representation


    def __call__(self, theta, xx):
        sigmaf = theta[0]
        l = theta[1]
        return self.f(sigmaf, l, xx)

    def value_and_grad(self, theta, xx):
        sigmaf = theta[0]
        l = theta[1]
        K = self.f(sigmaf, l, xx)
        dK = np.zeros((2,) + xx.shape)
        dK[0] = self.dfds(sigmaf, l, xx)
        dK[1] = self.dfdl(sigmaf, l, xx)
        return K, dK


