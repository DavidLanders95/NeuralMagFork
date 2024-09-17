import itertools

import matplotlib.pyplot as plt
import numpy as np

### plot eval
data = np.loadtxt("timings.dat")
single = np.loadtxt("ref/timings_single.dat")
double = np.loadtxt("ref/timings_double.dat")
magnumnp = np.loadtxt("ref/magnumnp.dat")
mumax = np.loadtxt("ref/mumax.dat")

fig, ax = plt.subplots(figsize=(15, 5))
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

ax.loglog(data[:, 0], 6 * data[:, 3] / 1e6, color=cycle[0], label="neuralmag(jax)")
ax.loglog(single[:, 0], single[:, 3] / 1e6, color=cycle[1], label="neuralmag(single)")
ax.loglog(
    double[:, 0],
    double[:, 3] / 1e6,
    "--",
    color=cycle[1],
    alpha=0.5,
    label="neuralmag(double)",
)
ax.loglog(magnumnp[:, 0], magnumnp[:, 3] / 1e6, color=cycle[2], label="magnum.np")
ax.loglog(mumax[:, 0], mumax[:, 1] / 1e6, "k--", label="MuMax3")

ax.set_xlabel("Number of Cells N")
ax.set_ylabel(r"Throughput [million cells / s]")
ax.legend()
ax.grid()
fig.savefig("timings.png")
