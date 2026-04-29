import celerite2
from celerite2 import terms
import numpy as np
from scipy.stats import norm
import sys
import os
from extract_data import *

# Directory containing mymodel.py
_here = os.path.dirname(os.path.abspath(__file__))

# Parent directory: NewCAR/
_parent = os.path.abspath(os.path.join(_here, ".."))

# Add parent to sys.path so we can import shared.py
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import shared
from shared import log_likelihood as _log_likelihood

num_qsos = 190
num_bands = 3
num_hyperparameters = 11
num_params = num_hyperparameters + 4*num_qsos*num_bands

# Load all the data
data = []
for i in range(num_qsos):
    for band in ["g", "r", "i"]:
        data.append(get_data(i, band, plot=False, sanitise=True))

log10_lbol = np.array([d["log10_lbol"] for d in data])
log10_lambda = np.array([d["log10_lambda"] for d in data])
z = np.array([d["redshift"] for d in data])
log10_1plusz = np.log10(1.0 + z)
mean_log10_lbol = np.mean(log10_lbol)
mean_log10_lambda = np.mean(log10_lambda)
mean_log10_1plusz = np.mean(log10_1plusz)


names = ["mu_mag", "sig_mag", "mu_log10_sigma", "sig_log10_sigma",
         "beta0", "beta1", "beta2", "n", "sig_log10_eta", "mu_log10_jitter",
         "sig_log10_jitter"]

def prior_transform(us):
    mu_mag  = 15.0 + 10.0*us[0]         # typical magnitude
    sig_mag = 5.0*us[1]                 # diversity of magnitudes
    mu_log10_sigma = -3.0 + 4.0*us[2]   # typical log10_sigma
    sig_log10_sigma = 5.0*us[3]         # diversity of log10_sigma
    
    # Regression parameters for log10_eta as response variable
    beta0 = -5.0 + 5.0*us[4]            # typical log10_eta
    beta1 = -10.0 + 20.0*us[5]          # log10_lambda slope
    beta2 = -10.0 + 20.0*us[6]          # log10_lbol slope
    n = -3.0 + 6.0*us[7]                # redshift slope
    sig_log10_eta = 5.0*us[8]           # intrinsic scatter
    mu_log10_jitter = -3.0 + 2.0*us[9]  # typical log10_jitter
    sig_log10_jitter = 5.0*us[10]       # diversity of log10_jitter

    qso_params_3d = us[num_hyperparameters:].copy()
    qso_params_3d = qso_params_3d.reshape((num_qsos, num_bands, 4))

    qso_params_3d[:, :, 0] = norm.ppf(qso_params_3d[:, :, 0],
                                      loc=mu_mag, scale=sig_mag)

    qso_params_3d[:, :, 1] = norm.ppf(qso_params_3d[:, :, 1],
                                      loc=mu_log10_sigma, scale=sig_log10_sigma)

    # Regression line prediction for log10_eta
    reg = beta0 + beta1*(log10_lambda - mean_log10_lambda) \
                + beta2*(log10_lbol - mean_log10_lbol) \
                + n*(log10_1plusz - mean_log10_1plusz)

    reg = reg.reshape((num_qsos, num_bands))
    qso_params_3d[:, :, 2] = norm.ppf(qso_params_3d[:, :, 2],
                                      loc=reg, scale=sig_log10_eta)
    qso_params_3d[:, :, 3] = norm.ppf(qso_params_3d[:, :, 3],
                                      loc=mu_log10_jitter,
                                      scale=sig_log10_jitter)

    hypers = np.array([mu_mag, sig_mag, mu_log10_sigma, sig_log10_sigma,
                       beta0, beta1, beta2, n, sig_log10_eta, mu_log10_jitter,
                       sig_log10_jitter])

    return np.hstack([hypers, qso_params_3d.flatten()])



def log_likelihood(params):

    qso_params_3d = params[num_hyperparameters:]\
                        .reshape((num_qsos, num_bands, 4))

    logl = 0.0
    k = 0
    for i in range(num_qsos):
        for j in range(3):
            logl += _log_likelihood(qso_params_3d[i, j, :],
                                    data[k]["light_curve"])
            k += 1

    return logl

def both(us):
    return log_likelihood(prior_transform(us))

