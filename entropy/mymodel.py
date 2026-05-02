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

# Indicate selections here
import shared
from shared import *

# Wrap log_likelihood
from shared import log_likelihood as _log_likelihood
def log_likelihood(params, data):
    return _log_likelihood(params, data, fixed_mean=False)

# Override prior_transform
prior_transform = prior_transform_informative

def generate_data(params):
    """
    Generate a dataset.
    """
    return shared.generate_light_curve(params)


def load_options(path="POSTENT_OPTIONS"):
    opts = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()   # remove comments
            if not line:
                continue
            if "=" not in line:
                continue
            key, val = map(str.strip, line.split("=", 1))
            try:
                # try float first (handles 1.0E-4)
                num = float(val)
                # but if it's actually an integer, convert to int
                if num.is_integer():
                    num = int(num)
                opts[key] = num
            except ValueError:
                # fallback: store raw string
                opts[key] = val
    return opts

opts = load_options()
tolerance = opts["tolerance"]

def log_kernel(params1, params2):
    """
    Distance to another particle in parameter space.
    """
    log10_tau1 = np.log10(2.0) + 2.0*(params1[1] - params1[2])
    log10_tau2 = np.log10(2.0) + 2.0*(params2[1] - params2[2])

    logp = -0.5*((log10_tau1 - log10_tau2)/tolerance)**2 \
            - 0.5*np.log(2.0*np.pi*tolerance**2)
    return logp

