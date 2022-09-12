import numpy as np
from smoove.utils import kanterp, kanterp3



def test_kanterp():
    def func(x):
        return 10*np.sin(20*x) #*np.exp(-x**2/0.25) + np.exp(x)

    # np.random.seed(420)

    N = 128
    x = np.sort(np.random.random(N))
    xp = np.linspace(0, 1, 100)
    f = func(x)
    ft = func(xp)
    sigman = np.ones(N)
    # sigman = np.exp(np.random.randn(N)) #/10000
    n = sigman*np.random.randn(N)
    w = 1/sigman**2
    y = f + n

    # # add outliers
    # for i in range(int(0.1*N)):
    #     idx = np.random.randint(0, N)
    #     y[idx] += 10 * np.random.randn()
    #     w[idx] = 0.25

    ms, Ps = kanterp(x, y, w, 10, nu0=3)

    diff = f - ms[0, :]
    sigma =  np.sqrt(Ps[0, 0, :])

    # import pdb; pdb.set_trace()

    Iin = np.abs(diff) < sigma

    print(Iin.sum()/Iin.size)

test_kanterp()


