import numpy as np
import scipy.special as scs
import numpy.linalg as lg
import scipy.special as scs


def lp1(x, nu, w):
    """Returns log p(x | v, w).
    Input:
    x: double
    v: np.array, shape: (K,)
    w: np.array, shape: (K,)
    """
    K = len(w)

    # Change all to logs and sum/sub instead of mult and div log(a**b) = b*log(a)
    arr = [np.log(w[k]) + np.log(2) + (nu[k]*np.log(nu[k])) + ((2*nu[k]-1)
                                                               * np.log(x)) + (-nu[k]*x**2) - scs.gammaln(nu[k]) for k in range(K)]

    # Log sum exp, have an implementation below in case the scipy is not allowed
    return scs.logsumexp(arr)


def log_sum_exp(arr):
    # In case using special.logsumexp is not allowed, here is an implementation

    max_val = np.max(arr)
    return max_val + np.log(np.sum(np.exp(arr - max_val)))


def lp2(x, Psi, nu):
    """Returns log p(x | \Psi, \nu)
    Input:
    x: np.array, shape: (p,p)
    Psi: np.array, shape: (p,p)
    nu: double, \nu > p-1
    """
    p = len(x)
    _, log_det_x = lg.slogdet(x)
    _, log_det_Psi = lg.slogdet(Psi)

    # Decompose with cholesky, and calcuate the trace using einsum (hopefuly correctly)
    Y = np.linalg.solve(np.linalg.cholesky(x), Psi)
    tr = np.einsum('ij,ji', Y, Y.T)

    return (0.5 * nu * log_det_Psi
            - 0.5 * nu * p * np.log(2)
            - scs.multigammaln(nu / 2, p)
            - 0.5 * (nu+p+1) * log_det_x
            - 0.5 * tr)

# x = np.linspace(1e-20, 5, 1000)
# nu = np.array([1,2])
# w = np.array([0.4, 0.6])

# test1 = [lp1(x, nu, w) for x in x]
# print(test1)
