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
plt.plot(truths[:,0], diffs, ".", alpha=0.3)
plt.show()
