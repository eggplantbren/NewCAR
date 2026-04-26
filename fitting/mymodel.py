import celerite2
from celerite2 import terms
import numpy as np
from scipy.stats import norm
import sys
import os

# Directory containing mymodel.py
_here = os.path.dirname(os.path.abspath(__file__))

# Parent directory: NewCAR/
_parent = os.path.abspath(os.path.join(_here, ".."))

# Add parent to sys.path so we can import shared.py
if _parent not in sys.path:
    sys.path.insert(0, _parent)


data = np.loadtxt("data.txt")
import shared
num_params = shared.num_params
prior_transform = shared.prior_transform

fix_mu = False
mean_mag = np.sum(data[:,1]/data[:,2]**2) \
            / np.sum(1.0/data[:,2]**2)

def log_likelihood(params):
    return shared.log_likelihood(params, data)


def both(us):
    return log_likelihood(prior_transform(us))

