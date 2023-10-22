# Write your functions lp1() and lp2() to replace the functions here. Do not change the names of the functions.
# Then copy the content of this cell into a separate file named 'problem2.py' to be submitted separately on Moodle.
# The file should include these import instructions and no others.
# Note that 'problem2.py' should be a standard Python source file (i.e., a text file, not a Jupyter notebook).

import numpy as np
import numpy.linalg as lg
import scipy.linalg as slg
import scipy.special as scs
import matplotlib.pyplot as plt
from scipy.special import multigammaln
from scipy import linalg
from numpy.linalg import slogdet

def lp1(x, nu, w):
    """Returns log p(x | v, w).
    Input:
    x: double
    v: np.array, shape: (K,)
    w: np.array, shape: (K,)
    """    
    K = len(w)

    # Change all to logs and sum/sub instead of mult and div
    # log(a**b) = b*log(a)
    arr = [np.log(w[k]) + np.log(2) + (nu[k]*np.log(nu[k])) + ((2*nu[k]-1) * np.log(x)) + (-nu[k]*x**2) -  scs.gammaln(nu[k]) for k in range(K)]

    # Log sum exp, have an implementation below in case the scipy is not allowed
    return scs.logsumexp(arr)

# In case using scipy is not allowed
def log_sum_exp(arr):
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
    _, log_det_x = slogdet(x)
    _, log_det_Psi = slogdet(Psi)

    # Decompose with cholesky, and calcuate the trace using einsum (hopefuly correctly)
    Y = np.linalg.solve(np.linalg.cholesky(x), Psi)
    tr = np.einsum('ij,ji->', Y, Y.T)

    return (0.5 * nu * log_det_Psi
        - 0.5 * nu * p * np.log(2)
        - multigammaln(nu / 2, p)
        - 0.5 * (nu+p+1) * log_det_x
        - 0.5 * tr)

# x = np.linspace(1e-20, 5, 1000)
# nu = np.array([0.1,2])
# w = np.array([0.9, 0.1])

# test1 = [lp1(x, nu, w) for x in x]

# plt.plot(x, test1)
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt


# def plot_log_inverse_wishart(Psi, nu, num_points=100):
#     """
#     Generate a plot of log-probability densities from the inverse Wishart distribution.

#     Parameters:
#     Psi : ndarray
#         Scale matrix of the inverse Wishart distribution.
#     nu : float
#         Degrees of freedom of the inverse Wishart distribution.
#     num_points : int, optional
#         Number of matrices to generate for the plot.
#     """
#     p = Psi.shape[0]
#     log_probs = []
#     determinants = np.logspace(-10, -1, num_points)  # Small determinant values

#     for det in determinants:
#         # Create a matrix with a given determinant and some randomness
#         A = np.random.rand(p, p)
#         Q, _ = np.linalg.qr(A)  # Make it orthogonal
#         X = Q @ np.diag([det] * p) @ Q.T

#         # Ensure it's symmetric positive definite
#         X = (X + X.T) / 2 + p * np.eye(p)

#         # Compute log-probability density
#         log_prob = lp2(X, Psi, nu)
#         log_probs.append(log_prob)

#     plt.plot(determinants, log_probs, marker='o', linestyle='-')
#     plt.title('Log-Probability Density of Inverse Wishart Distribution')
#     plt.xlabel('Determinant of X')
#     plt.ylabel('Log-Probability Density')
#     plt.xscale('log')
#     plt.show()

# # Define symmetric positive definite matrix Psi and degrees of freedom nu
# p = 50  # Size of the matrix
# Psi = np.eye(p)  # Identity matrix as an example
# nu = p + 1  # Degrees of freedom (should be > p - 1)

# # Generate plot
# plot_log_inverse_wishart(Psi, nu)
