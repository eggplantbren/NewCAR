import numpy as np
import matplotlib.pyplot as plt
import corner

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})


ndim = 4

def cornerplot(posterior_sample):
    mu = posterior_sample[:,0]
    log10_sigma = posterior_sample[:,1]
    log10_beta = posterior_sample[:,2]
    log10_jitter = posterior_sample[:,3]
    log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

    posterior_sample = np.column_stack((mu, log10_beta, log10_tau, log10_jitter))

    figure = corner.corner(posterior_sample,
        labels=["$\\mu$", "$\\log_{10}(\\beta)$",
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
log10_beta = posterior_sample[:,2]
log10_jitter = posterior_sample[:,3]
log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

plt.hist(log10_tau, 40, density=True, alpha=0.3,
         label="Free $\\mu$")
plt.title("Flat Priors")
plt.xlim([1.0, 10.0])
plt.axvline(3.60749121, color="k", alpha=0.6, label="True value")
plt.ylabel("Probability Density")

posterior_sample = np.loadtxt("results/posterior_sample_flat_fixed.txt")

mu = posterior_sample[:,0]
log10_sigma = posterior_sample[:,1]
log10_beta = posterior_sample[:,2]
log10_jitter = posterior_sample[:,3]
log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

plt.hist(log10_tau, 40, density=True, alpha=0.3,
         label="Fixed $\\mu$")
plt.legend()

plt.subplot(1, 2, 2)
posterior_sample = np.loadtxt("results/posterior_sample_informative_free.txt")

mu = posterior_sample[:,0]
log10_sigma = posterior_sample[:,1]
log10_beta = posterior_sample[:,2]
log10_jitter = posterior_sample[:,3]
log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

plt.hist(log10_tau, 40, density=True, alpha=0.3,
         label="Free $\\mu$")
plt.title("Informative Priors")


posterior_sample = np.loadtxt("results/posterior_sample_informative_fixed.txt")

mu = posterior_sample[:,0]
log10_sigma = posterior_sample[:,1]
log10_beta = posterior_sample[:,2]
log10_jitter = posterior_sample[:,3]
log10_tau = (log10_sigma - log10_beta + 0.5*np.log10(2))*2

plt.hist(log10_tau, 40, density=True, alpha=0.3,
         label="Fixed $\\mu$")
plt.title("Informative Priors")


plt.xlabel("$\\log_{10}(\\tau/{\\rm days})$")
plt.xlim([1.0, 10.0])
plt.axvline(3.60749121, color="k", alpha=0.6, label="True value")
plt.legend()

plt.savefig("four_posteriors.pdf")
plt.show()

