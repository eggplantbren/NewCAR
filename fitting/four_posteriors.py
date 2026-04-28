import numpy as np
import matplotlib.pyplot as plt
import dnest4

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

