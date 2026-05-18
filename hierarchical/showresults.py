import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt
import corner
from mymodel import names

dn4.postprocess()

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
})


posterior_sample = np.loadtxt("posterior_sample.txt")

# Subset
hypers = posterior_sample[:, 4:9]
ndim = hypers.shape[1]

figure = corner.corner(hypers,
    labels=[r"$\beta_0$", r"$\beta_1$", r"$\beta_2$",
            r"$n$", r"$s_{\log_{10}(\eta)}$"],
            plot_contours=False,
            plot_density=False, fontsize=14,
            hist_kwargs={"color":"blue", "alpha":0.2, "histtype":"stepfilled",
                         "edgecolor":"black","lw":"3"} )

axes = np.array(figure.axes).reshape((ndim, ndim))

for i in range(ndim):
    ax = axes[i, i]

plt.savefig("hierarchical_corner.png", dpi=400)
plt.show()

