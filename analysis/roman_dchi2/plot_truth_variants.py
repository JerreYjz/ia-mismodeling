#!/usr/bin/env python3
"""Roman Delta chi2 (P(k)-fit amplitudes, IR/UV mask at k_UV = k_max and
k0/D(z) family) for the three truth variants: highk, nla, bconst."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
LS = {"highk": "-", "nla": "--", "bconst": ":"}
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for t in LS:
    R = np.load(f"results_{t}/roman_dchi2.npz")
    Fm = np.load(f"results_{t}/family_noise.npz")
    for m in COL:
        axs[0].plot(R["kmax"], R[f"{m}_diag"], LS[t], marker="o", ms=3, color=COL[m], label=f"{m}, truth {t}")
        ok = Fm[m] > 0
        axs[1].plot(Fm["k0"][ok], Fm[m][ok], LS[t], marker="o", ms=3, color=COL[m])
for ax, xl in zip(axs, [r"$k_{\rm max}=k_{UV}$ (constant comoving)", r"$k_0$ of $k_{UV}=k_0/D(z)$"]):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(1, color="k", lw=0.8); ax.axhline(0.2, color="k", lw=0.8, ls="--")
    ax.set_xlabel(xl + r" $[h\,{\rm Mpc}^{-1}]$"); ax.set_ylabel(r"Roman $\Delta\chi^2$ (P(k)-fit amplitudes)")
axs[0].legend(fontsize=7)
fig.suptitle("Sensitivity to the FLAMINGO truth: shape-noise subtraction (highk / nla) and bias convention (bconst)")
fig.tight_layout(); fig.savefig("results_highk/truth_variants.png", dpi=120)
