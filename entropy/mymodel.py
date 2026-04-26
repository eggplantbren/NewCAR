import numpy as np
import numpy.random as rng
from scipy.special import gammaln
from scipy.stats import norm
import celerite2
from celerite2 import terms
import matplotlib.pyplot as plt
import sys
import os

# Directory containing mymodel.py
_here = os.path.dirname(os.path.abspath(__file__))

# Parent directory: NewCAR/
_parent = os.path.abspath(os.path.join(_here, ".."))

# Add parent to sys.path so we can import shared.py
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import shared
num_params = shared.num_params
prior_transform = shared.prior_transform
log_likelihood = shared.log_likelihood


rng.seed(1234)

days = 365*100
window = np.arange(15000, 15000+20*365)
err = 0.015
n = 250
t = np.arange(days)
choice = rng.choice(window, size=n, replace=False)
choice = np.sort(choice)
t_obs = t[choice]

def generate_light_curve(params):
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

    return data

def generate_data(params):
    """
    Generate a dataset.
    """
    return generate_light_curve(params)

def distance(params1, params2):
    """
    Distance to another particle in parameter space.
    """
    log10_tau1 = np.log10(2.0) + 2.0*(params1[1] - params1[2])
    log10_tau2 = np.log10(2.0) + 2.0*(params2[1] - params2[2])

    return np.abs(log10_tau1 - log10_tau2)

