import numpy as np
import numpy.random as npr
import scipy.stats as sps
import matplotlib.pyplot as plt

def ltarget(x):
    if np.abs(x)>.99 and np.abs(x) < 3:
        return sps.norm.logpdf(x, 0, 1)
    else:
        return -np.inf

def eval_logq(xp, x):
    return 0

def sample_q(x):
    return x + npr.triangular(-5,0,5)

x0 = 0
x = 0
samples = [x]
accepted_count = 0

for i in range(10000):     
    xp = sample_q(x)
    accrate = np.minimum(1, np.exp(ltarget(xp) + eval_logq(x, xp) - ltarget(x) - eval_logq(xp, x)))
    if (npr.uniform() < accrate):
        x = xp
        accepted_count += 1

    samples.append(x)

print(f'Acceptrance rate is {accepted_count / len(samples)}')
samples = np.array(samples[len(samples)//2:])
# plt.plot(np.arange(len(samples)), samples)
x=np.linspace(-20, 20)
plt.plot(x, [ltarget(x) for x in x])
plt.hist(samples, bins=100, density=True)
plt.show()
