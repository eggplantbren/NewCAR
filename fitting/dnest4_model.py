import celerite2
from celerite2 import terms
import numpy as np
from scipy.stats import norm

data = np.loadtxt("data.txt")
num_params = 4

fix_mu = False
mean_mag = np.sum(data[:,1]/data[:,2]**2) \
            / np.sum(1.0/data[:,2]**2)

def prior_transform(us):
    params = us.copy()

    # (mu, log10_sigma, log10_beta, log10_jitter)
    params[0] = 20.0 + 1.5*norm.ppf(us[0])
    params[1] = -0.5 + 0.5*norm.ppf(us[1])
    params[2] = -2.0 + 0.6*norm.ppf(us[2])
    params[3] = -2.0 + 0.5*norm.ppf(us[3])

    return params

def log_likelihood(params):

    logl = 0.0

    mu = params[0]
    sigma, beta, jitter = 10.0**params[1:4]
    tau = 2*(sigma/beta)**2

    try:
        term = terms.RealTerm(a=sigma**2, c=1.0/tau)
        kernel = term
        gp = celerite2.GaussianProcess(kernel, mean=mu)
        gp.compute(data[:,0], yerr=np.sqrt(data[:,2]**2 + jitter**2))
        logl = gp.log_likelihood(data[:,1])
    except Exception:
        logl = -1.0E300

    return logl


def both(us):
    return log_likelihood(prior_transform(us))

