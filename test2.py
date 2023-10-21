import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from scipy.special import expit

heart_data = pd.read_csv('./heart_data.txt', sep='\t')
# Extract input variables
p4_x = heart_data[['SBP', 'DBP', 'CIG', 'AGE']].values.astype(float)
# Extract target variable
p4_y = heart_data['HAS_CHD'].values.astype(float)
# Remove mean from inputs
p4_x -= np.mean(p4_x, 0)
# Standardise input variance
p4_x /= 2*np.std(p4_x, 0)

np.random.seed(1)

def logistic(x, beta0, beta):
    return 1 / (1 + np.exp(-beta0 - np.dot(x, beta)))

def log_likelihood(beta0, beta, x, y):
    h = logistic(x, beta0, beta)

    return np.sum(np.log(h**y * (1-h)**(1-y)))

def log_prior(beta0, beta, log_sigma_beta):
    sigma_beta = np.exp(log_sigma_beta) 

    beta_log_prior = np.log(norm.pdf(beta0, 0, sigma_beta)) + np.sum(np.log(norm.pdf(beta, 0, sigma_beta)))
    sigma_prior = chi2.pdf(sigma_beta, df=1) + sigma_beta
    return beta_log_prior + np.log(sigma_prior)

def log_posterior(beta0, beta, log_sigma_beta, x, y):
    return log_likelihood(beta0, beta, x, y) + log_prior(beta0, beta, log_sigma_beta)

def proposal(current):
    # Adjusted proposal cov until the acceptance rate is good enough
    return np.random.multivariate_normal(current, 0.002*np.eye(len(current)))

def mcmc(initial, N, x, y):
    current = initial
    samples = [current]
    accept_count = 0
    for i in range(N):
        proposed = proposal(current)
        p_log_sigma_beta = proposed[0]
        p_beta0 = proposed[1]
        p_beta = proposed[2:6]

        log_sigma_beta = current[0]
        beta0 = current[1]
        beta = current[2:6]

        a = log_posterior(p_beta0, p_beta , p_log_sigma_beta, x, y) - log_posterior(beta0, beta, log_sigma_beta, x, y)
        
        if np.random.rand() < np.exp(a):
            current = proposed
            accept_count+=1
        
        samples.append(current)
    
    print(f'Acceptance rate={accept_count/N}')
    return np.array(samples)

# Initialize the parameters and run the sampler
initial = np.array([0.5, 4, 4, 4, 4, 4]) # [sigma_beta, beta0, beta1, beta2, beta3, beta4]
samples = mcmc(initial, 5000, p4_x, p4_y)
samples = np.array(samples[len(samples)//2:])

parameter_names = ['log_sigma_beta', 'beta0', 'beta1', 'beta2', 'beta3', 'beta4']

for i in range(len(samples[0])):
    plt.plot(np.arange(len(samples)), samples[:,i], label=parameter_names[i])

means = np.mean(samples, axis=0)

posterior_median = means # do not hard-code the number: assign the variables used in your previous computations
posterior_lower = np.percentile(samples, 25, axis=0) # do not hard-code the number: assign the variables used in your previous computations
posterior_upper =  np.percentile(samples, 75, axis=0) # do not hard-code the number: assign the variables used in your previous computations

print('Posterior medians (50% quantiles): {}'.format(posterior_median))
print('Posterior 25% quantiles: {}'.format(posterior_lower))
print('Posterior 75% quantiles: {}'.format(posterior_upper))

plt.legend()
plt.show()
samples_subset = samples[-1000:]

# # Create the scatter plots
# num_params = len(samples_subset[0])
# fig, ax = plt.subplots(num_params, num_params, figsize=(12, 12))

# for i in range(num_params):
#     for j in range(num_params):
#         if i == j:
#             ax[i, j].hist(samples_subset[:, i], bins=50, color='gray', alpha=0.6)
#         else:
#             ax[i, j].scatter(samples_subset[:, j], samples_subset[:, i], alpha=0.6, s=5)
        
#         if i == num_params - 1:
#             ax[i, j].set_xlabel(parameter_names[j])
#         if j == 0:
#             ax[i, j].set_ylabel(parameter_names[i])

# plt.tight_layout()
# plt.show()

# means = np.mean(samples, axis=0)
