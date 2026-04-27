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

def generate_data(params):
    """
    Generate a dataset.
    """
    return shared.generate_light_curve(params)

def distance(params1, params2):
    """
    Distance to another particle in parameter space.
    """
    log10_tau1 = np.log10(2.0) + 2.0*(params1[1] - params1[2])
    log10_tau2 = np.log10(2.0) + 2.0*(params2[1] - params2[2])

    return np.abs(log10_tau1 - log10_tau2)

