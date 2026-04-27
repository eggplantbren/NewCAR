# Stuff shared across fitting and entropy

import numpy as np
import numpy.random as rng
import celerite2
from scipy.stats import norm
from celerite2 import terms

num_params = 4

days = 365*100
window = np.arange(15000, 15000+20*365)
err = 0.015
n = 250
t = np.arange(days)
choice = rng.choice(window, size=n, replace=False)
choice = np.sort(choice)
t_obs = t[choice]

def generate_light_curve(params, return_y=False):
    mu, log10_sigma, log10_beta, log10_jitter = params
    sigma = 10.0**log10_sigma
    beta = 10.0**log10_beta
    jitter = 10.0**log10_jitter
    tau = 2*(sigma/beta)**2

    alpha = np.exp(-1.0/tau)
    eps = sigma*np.sqrt(1.0 - alpha**2)

    y = np.empty(days)
    y[0] = mu + sigma*rng.randn()
    for i in range(1, days):
        y[i] = mu + alpha*(y[i-1] - mu) + eps*rng.randn()

    y_obs = y[choice] + np.sqrt(jitter**2 + err**2)*rng.randn(n)
    err_obs = err*np.ones(n)

    # Save the data
    data = np.column_stack((t_obs, y_obs, err_obs))

    if return_y:
        return [data, y]
    return data


def log_likelihood(params, data):

    logl = 0.0

    mu = params[0]
    sigma, beta, jitter = 10.0**params[1:4]
    tau = 2*(sigma/beta)**2

    #mu = np.sum(data[:,1]/data[:,2]**2)/np.sum(1.0/data[:,2]**2)

    try:
        term = terms.RealTerm(a=sigma**2, c=1.0/tau)
        kernel = term
        gp = celerite2.GaussianProcess(kernel, mean=mu)
        gp.compute(data[:,0], yerr=np.sqrt(data[:,2]**2 + jitter**2))
        logl = gp.log_likelihood(data[:,1])
    except Exception:
        logl = -1.0E300

    return logl


means = np.array([20.0, -0.5, -2.0, -2.0])
sds = np.array([1.5, 0.5, 0.6, 0.5])
def prior_transform(us):
    params = us.copy()

    # (mu, log10_sigma, log10_beta, log10_jitter)
    #return means + sds*norm.ppf(us)

    # Faster, logistic approximation
    return means + 0.55*sds*np.log(us/(1.0 - us))


