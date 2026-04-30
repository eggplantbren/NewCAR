import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import corner

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})

rng.seed(123)
ndim = 4


def cornerplot(posterior_sample):
    mu = posterior_sample[:,0]
    log10_sigma = posterior_sample[:,1]
    log10_eta = posterior_sample[:,2]
    log10_jitter = posterior_sample[:,3]
    log10_tau = (log10_sigma - log10_eta + 0.5*np.log10(2))*2

    posterior_sample = np.column_stack((mu, log10_eta, log10_tau, log10_jitter))

    figure = corner.corner(posterior_sample,
        labels=["$\\mu$", "$\\log_{10}(\\eta)$",
                "$\\log_{10}(\\tau)$",
                "$\\log_{10}({\\rm jitter})$"],
                plot_contours=False,
                plot_density=False, fontsize=14,
                hist_kwargs={"color":"blue", "alpha":0.2, "histtype":"stepfilled",
                             "edgecolor":"black","lw":"3"} )

    axes = np.array(figure.axes).reshape((ndim, ndim))

    for i in range(ndim):
	    ax = axes[i,i]


cornerplot(np.loadtxt("results/posterior_sample_informative_free.txt"))
plt.savefig("cornerplot.pdf", bbox_inches="tight")
plt.show()


cornerplot(np.loadtxt("results/posterior_sample_flat_free.txt"))
plt.savefig("cornerplot_flat.pdf", bbox_inches="tight")
plt.show()


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
posterior_sample = np.loadtxt("results/posterior_sample_flat_free.txt")

mu = posterior_sample[:,0]
log10_sigma = posterior_sample[:,1]
log10_eta = posterior_sample[:,2]
log10_jitter = posterior_sample[:,3]
log10_tau = (log10_sigma - log10_eta + 0.5*np.log10(2))*2

plt.hist(log10_tau, 40, density=True, alpha=0.3,
         label="Free $\\mu$")
plt.title("Flat Priors")
plt.xlim([1.0, 10.0])
plt.xlabel("$\\log_{10}(\\tau/{\\rm days})$")
plt.ylabel("Probability Density")

posterior_sample = np.loadtxt("results/posterior_sample_flat_fixed.txt")

mu = posterior_sample[:,0]
log10_sigma = posterior_sample[:,1]
log10_eta = posterior_sample[:,2]
log10_jitter = posterior_sample[:,3]
log10_tau = (log10_sigma - log10_eta + 0.5*np.log10(2))*2

plt.hist(log10_tau, 40, density=True, alpha=0.3,
         label="Fixed $\\mu$")

plt.axvline(3.7787833316317925, color="k", alpha=0.6, label="True value")
plt.legend()

plt.subplot(1, 2, 2)
posterior_sample = np.loadtxt("results/posterior_sample_informative_free.txt")

mu = posterior_sample[:,0]
log10_sigma = posterior_sample[:,1]
log10_eta = posterior_sample[:,2]
log10_jitter = posterior_sample[:,3]
log10_tau = (log10_sigma - log10_eta + 0.5*np.log10(2))*2

plt.hist(log10_tau, 40, density=True, alpha=0.3,
         label="Free $\\mu$")
plt.title("Informative Priors")


posterior_sample = np.loadtxt("results/posterior_sample_informative_fixed.txt")

mu = posterior_sample[:,0]
log10_sigma = posterior_sample[:,1]
log10_eta = posterior_sample[:,2]
log10_jitter = posterior_sample[:,3]
log10_tau = (log10_sigma - log10_eta + 0.5*np.log10(2))*2

plt.hist(log10_tau, 40, density=True, alpha=0.3,
         label="Fixed $\\mu$")
plt.title("Informative Priors")


plt.xlabel("$\\log_{10}(\\tau/{\\rm days})$")
plt.xlim([1.0, 10.0])
plt.axvline(3.7787833316317925, color="k", alpha=0.6, label="True value")
plt.legend()

plt.savefig("four_posteriors.pdf")
plt.show()


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

n = 1000000
params = []
for i in range(n):
    params.append(prior_transform_informative(rng.rand(num_params)))
params = np.vstack(params)

log10_sigma = params[:,1]
log10_eta   = params[:,2]
log10_tau   = np.log10(2.0) + 2.0*(log10_sigma - log10_eta)
plt.hist(log10_tau, density=True, bins=100, alpha=0.3, label="Informative Prior")
plt.xlabel("$\\log_{10}(\\tau)$")
plt.ylabel("Prior Density")
print(np.mean(log10_tau), np.std(log10_tau))

params = []
for i in range(n):
    params.append(prior_transform_flat(rng.rand(num_params)))
params = np.vstack(params)

log10_sigma = params[:,1]
log10_eta   = params[:,2]
log10_tau   = np.log10(2.0) + 2.0*(log10_sigma - log10_eta)
plt.hist(log10_tau, density=True, bins=100, alpha=0.3, label="Flat Prior")
plt.xlabel("$\\log_{10}(\\tau)$")
plt.ylabel("Prior Density")
print(np.mean(log10_tau), np.std(log10_tau))




plt.legend()
plt.savefig("tau_priors.pdf")
plt.show()

