import numpy as np
import numpy.random as npr
import scipy.stats as sps
import matplotlib.pyplot as plt

def ltarget(x):
        return -np.log(np.abs(np.sin(x)))

def eval_logq(xp, x):
    return sps.norm.logpdf(xp,loc=x,scale=1.0)

def sample_q(x):
    return x + npr.normal(0, 1.0)

x = 5
samples = [x]
accepted_count = 0

for i in range(1000):
    xp = sample_q(x)
    a = np.minimum(1, np.exp(ltarget(xp) + eval_logq(x, xp) - ltarget(x) - eval_logq(xp, x)))
    if (npr.uniform() < a):
        x = xp
        accepted_count += 1

    samples.append(x)

print(f'Acceptrance rate is {accepted_count / len(samples)}')
samples = np.array(samples)[len(samples)//2:]
# plt.plot(np.arange(len(samples)), samples)
x=np.linspace(-5000, 5000, 500)
plt.plot(x, [np.exp(ltarget(x)) for x in x])
plt.hist(samples, bins=300, density=True)
plt.show()
