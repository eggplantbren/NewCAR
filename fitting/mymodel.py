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
from shared import *
from shared import log_likelihood as _log_likelihood


import json

with open("config.json") as f:
    cfg = json.load(f)

fixed_mean = cfg["fixed_mean"]
prior = cfg["prior"]

prior_transform = None
if prior == "informative":
    prior_transform = prior_transform_informative
elif prior == "flat":
    prior_transform = prior_transform_flat

def log_likelihood(params):
    return _log_likelihood(params, data, fixed_mean=fixed_mean)

def both(us):
    return log_likelihood(prior_transform(us))

