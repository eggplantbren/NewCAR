import numpy as np
import matplotlib.pyplot as plt

logzs  = np.atleast_1d(np.loadtxt("logzs.txt"))
logzs2 = np.atleast_1d(np.loadtxt("logzs2.txt"))

# Truncate to equal length
logzs = logzs[0:len(logzs2)]

# Calculate entropy
diffs = logzs - logzs2
print(f"Number of runs = {len(diffs)}.")
mean = np.mean(diffs)
sd   = np.std(diffs, ddof=1)
sem  = sd/np.sqrt(len(diffs))
print(f"Mean diff = {mean}, SD of diffs = {sd}.")
print(f"H = {mean} +- {sem}.")

truths = np.loadtxt("truths.txt")
if truths.ndim == 1:
    truths = truths[:, None]
truths = truths[0:len(diffs),:]

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})

log10_tau = np.log(2.0) + 2.0*(truths[:,1] - truths[:,2])
plt.plot(log10_tau, diffs, ".", alpha=0.3)
plt.xlabel("$\\log_{10}(\\tau)$")
plt.ylabel("Estimated $-\\log p(\\log_{10}(\\tau) \\,|\\, {\\rm data})|_{\\rm truth}$")
plt.axvline(np.log10(1.0), linewidth=2, linestyle="--", alpha=0.2, color="k")
plt.axvline(np.log10(20.0*365), linewidth=2, linestyle="--", alpha=0.2, color="k")
plt.show()
