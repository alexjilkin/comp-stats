import numpy as np
from scipy.stats import gamma
from scipy.special import gamma as gamma_func
import matplotlib.pyplot as plt

# Target distribution
def p(theta):
    return np.sqrt(2/np.pi) * (theta**2 * np.exp(-theta**2 / 8) / 8)

# Proposal distribution: Gamma distribution
a, scale = 2, 2  # example parameters, you may need to adjust these
def q(theta):
    return gamma.pdf(theta, a=a, scale=scale)

# Number of samples
N = 50000

x = np.linspace(0, 50, 1000)
plt.plot(x, p(x))
plt.plot(x, q(x))
plt.show()

# Importance Sampling
samples = gamma.rvs(a=a, scale=scale, size=N)
weights = p(samples) / q(samples)
V_est = np.mean(weights)
V_std = np.std(weights)

# 95% Confidence Interval
conf_int = [V_est - 1.96 * V_std / np.sqrt(N), V_est + 1.96 * V_std / np.sqrt(N)]

print("Estimated V:", V_est)
print("95% Confidence Interval:", conf_int)
