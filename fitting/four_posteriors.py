import numpy as np
import matplotlib.pyplot as plt
import dnest4
import config
import os

def run_cpp(seed):
    os.system(f"./main -s {seed}")
    dnest4.postprocess(plot=False, rng_seed=123)

# Run 1
config.fixed_mean = False
config.prior = "informative"
run_cpp(seed=1)
os.system("mv posterior_sample.txt posterior_sample1.txt")

# Run 2
config.fixed_mean = True
config.prior = "informative"
run_cpp(seed=2)
os.system("mv posterior_sample.txt posterior_sample2.txt")

# Run 3
config.fixed_mean = False
config.prior = "flat"
run_cpp(seed=3)
os.system("mv posterior_sample.txt posterior_sample3.txt")

# Run 4
config.fixed_mean = True
config.prior = "flat"
run_cpp(seed=4)
os.system("mv posterior_sample.txt posterior_sample4.txt")

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.size": 14,
#})

#plt.figure(figsize=(14, 6))

#plt.subplot(1, 2, 1)
#posterior_sample = np.loadtxt("posterior_sample_flat.txt")

#mu = posterior_sample[:,0]
#log10_sigma = posterior_sample[:,1]
#log10_beta = posterior_sample[:,2]
#log10_jitter = posterior_sample[:,3]
#log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

#plt.hist(log10_tau, 40, density=True, alpha=0.3,
#         label="Free $\\mu$")
#plt.title("Flat Priors")
#plt.xlim([1.0, 10.0])
#plt.axvline(3.60749121, color="k", alpha=0.6, label="True value")
#plt.legend()
#plt.ylabel("Probability Density")

#posterior_sample = np.loadtxt("posterior_sample.txt")

#mu = posterior_sample[:,0]
#log10_sigma = posterior_sample[:,1]
#log10_beta = posterior_sample[:,2]
#log10_jitter = posterior_sample[:,3]
#log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

#plt.hist(log10_tau, 40, density=True, alpha=0.3,
#         label="Fixed $\\mu$")

#plt.subplot(1, 2, 2)
#posterior_sample = np.loadtxt("posterior_sample_informative.txt")

#mu = posterior_sample[:,0]
#log10_sigma = posterior_sample[:,1]
#log10_beta = posterior_sample[:,2]
#log10_jitter = posterior_sample[:,3]
#log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

#plt.hist(log10_tau, 40, density=True, alpha=0.3,
#         label="Free $\\mu$")
#plt.title("Informative Priors")


#posterior_sample = np.loadtxt("posterior_sample_informative2.txt")

#mu = posterior_sample[:,0]
#log10_sigma = posterior_sample[:,1]
#log10_beta = posterior_sample[:,2]
#log10_jitter = posterior_sample[:,3]
#log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

#plt.hist(log10_tau, 40, density=True, alpha=0.3,
#         label="Fixed $\\mu$")
#plt.title("Informative Priors")


#plt.xlabel("$\\log_{10}(\\tau/{\\rm days})$")
#plt.xlim([1.0, 10.0])
#plt.axvline(3.60749121, color="k", alpha=0.6, label="True value")
#plt.legend()

#plt.savefig("two_posteriors.pdf")
#plt.show()

