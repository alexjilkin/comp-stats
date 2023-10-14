import autograd.numpy as np 
import matplotlib.pyplot as plt
from autograd import grad
from autograd.scipy import stats

sig = np.array([[1, 0.998], [0.998, 1]])

def H(theta, r):
    return -pi(theta) + (1/2)*(np.sum(r**2))

def pi(theta):
    return stats.multivariate_normal.logpdf(theta, mean=np.zeros(2), cov=sig)

grad_logpdf_target = grad(pi)
    
def leapfrog(theta, r, epsilon, L):
    for _ in range(L):
        r = r + (epsilon / 2) * grad_logpdf_target(theta)
        theta = theta + epsilon * r
        r = r + (epsilon / 2) * grad_logpdf_target(theta)
        
    return theta, r

acceptance_rates = []
x = np.linspace(0.001, 0.1, 100)
H_orig = H(np.array([0,0]), np.array([1,1/3]))

for eps in x:
    theta_new, r_new = leapfrog(np.array([0,0]), np.array([1,1/3]), eps, 10)
    
    H_new = H(theta_new, r_new)
    acceptance_rates.append(np.exp(H_orig - H_new))

index_below_60 = next((i for i, v in enumerate(acceptance_rates) if v < 0.6), None)
index_above_10 = len(acceptance_rates) - 1 - next((i for i, v in enumerate(reversed(acceptance_rates)) if v > 0.1), None)

print(f'smallest eps below 60% = {x[index_below_60]}')
print(f'largest eps above 10% = {x[index_above_10]}')
# plt.plot(x, acceptance_rates)

distances = []
theta_0 = np.array([0,0])
r_0 = np.array([1, 1/3])

theta_i = theta_0.copy()
r_i = r_0.copy()


for _ in range(500):
    # H_old = H(theta_i, r)
    theta_i, r_i = leapfrog(theta_i, r_i, 0.05, 1)
    # H_new = H(theta_i, r)
    distances.append(np.linalg.norm(theta_0-theta_i))
distances = distances
# print(distances)
plt.plot(np.arange(len(distances)), distances)
plt.show()