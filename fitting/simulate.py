# Generate a simulated dataset

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng

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
from shared import *

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12,
})

if __name__ == "__main__":
    rng.seed(67)
    params = prior_transform_informative(rng.rand(num_params))
    log10_tau = np.log10(2.0) + 2.0*(params[1] - params[2])
    print(f"True values = {params}. log10_tau = {log10_tau}.")
    [data, y] = generate_light_curve(params,
                                     return_y=True)

    np.savetxt("data.txt", data)

    # Two panels in the plot
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.errorbar(data[:,0], data[:,1], yerr=data[:,2],
                  fmt=".", label="Observations")
    plt.ylabel("Magnitude")

    plt.subplot(2, 1, 2)
    plt.plot(shared.t, y, "r-", alpha=0.5, label="Underlying curve")
    plt.errorbar(data[:,0], data[:,1], yerr=data[:,2],
                 fmt=".", label="Observations")

    plt.legend()
    plt.xlabel("Time $t$ (days)")
    plt.ylabel("Magnitude")

    # Save and display the plot
    plt.savefig("simulation.pdf", bbox_inches="tight")
    plt.show()

