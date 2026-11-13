#!/usr/bin/env python3
"""Add IA B modes to the cosmolike (E-mode only, sim path) xi_IA: python
Limber xi from P_BB (validated against cosmolike's native TATT B-mode term:
residual chi2 0.17 of 41, check_tatt_bb.py), for FLAMINGO (P_BB minus the
white level, ratio to P_NL held below 0.05, nothing above 17.7) and for each
model fit. Writes analysis/dchi2_masks_withBB.csv and fig_bmodes."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import roman_limber as RL
R = np.load("run/cosmolike_xi_ia.npz"); d = np.load("../v2/fits.npz"); cov = np.load("analysis/cov_gauss_d3a.npy")
zs, kd, kg = d["zs"], d["k"][0], R["kgrid"]
cosmo = RL.Cosmology(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, As=2.099e-9, mnu=0.06)
S = RL.RomanShear(cosmo, sys.argv[1])
lo = np.geomspace(1e-4, 0.05, 30, endpoint=False); hi = np.geomspace(kd[-1]*1.02, 2e3, 20)
m = kd >= 0.05; kk = np.concatenate([lo, kd[m], hi])
zero = lambda K, Z: 0*K
def xbb(RBB):
    t = np.array([np.concatenate([np.full(len(lo), r[m][0]), r[m], np.zeros(len(hi))]) for r in RBB])
    CE, CB = S.cl({"dE": zero, "EE": zero, "BB": RL.ratio_table(zs, kk, t)}, gg=False)
    return S.xi(CE, CB)
q = lambda v, mm: float(v[mm] @ np.linalg.solve(cov[np.ix_(mm, mm)], v[mm]))
bt = xbb(d["Rtrue"][:, 2])
M1, K1 = R["masks_n1"], R["K_n1"]
allm = np.ones(1080, bool)
rows = {}
with open("analysis/dchi2_masks_withBB.csv", "w") as fh:
    fh.write("model,kmax,all_E,all_EB," + ",".join(f"n1_{k:g}_E,n1_{k:g}_EB" for k in K1) + "\n")
    for mdl in ("NLA", "TATT"):
        for j in range(len(kg)):
            bm = xbb(d[f"{mdl}_R"][:, j, 2])
            dE = R[f"xi_{mdl}_{j}"] - R["xi_flam"]
            dEB = dE + bm - bt
            r = [q(dE, allm), q(dEB, allm)]
            for mm in M1:
                r += [q(dE, mm), q(dEB, mm)]
            rows[(mdl, j)] = r
            fh.write(f"{mdl},{kg[j]:.3f}," + ",".join(f"{v:.4g}" for v in r) + "\n")
print("FLAMINGO B-mode xi alone: chi2 on 0.35/D mask = %.2f" % q(bt, M1[list(K1).index(0.35)]))
i35 = list(K1).index(0.35)
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
for ax, (lab, kk2) in zip(axs, (("0.35/D(z) mask", 2 + 2*i35), ("0.25/D(z) mask", 2 + 2*list(K1).index(0.25)))):
    for mdl, col in (("NLA", "#d1495b"), ("TATT", "#1f5fa8")):
        ax.plot(kg, [rows[(mdl, j)][kk2] for j in range(len(kg))], "o-", color=col, label=f"{mdl}: E modes only (cosmolike sim path)")
        ax.plot(kg, [rows[(mdl, j)][kk2 + 1] for j in range(len(kg))], "s--", color=col, label=f"{mdl}: + IA B modes (FLAMINGO and model)")
    ax.axhline(1, color="k", lw=1); ax.axhline(0.2, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.1, 300)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.12, 0.2, 0.3, 0.5, 1, 2, 3]))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g")); ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(r"fit $k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$"); ax.set_ylabel(r"$\Delta\chi^2$ (model $-$ FLAMINGO $\xi^{\rm IA}$)")
    ax.set_title(lab); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("analysis/fig_bmodes.png", dpi=220); fig.savefig("analysis/fig_bmodes.pdf"); plt.close(fig)
for mdl in ("NLA", "TATT"):
    print(mdl, "0.35/D  E:", " ".join(f"{rows[(mdl, j)][2 + 2*i35]:6.2f}" for j in range(len(kg))))
    print(mdl, "0.35/D EB:", " ".join(f"{rows[(mdl, j)][3 + 2*i35]:6.2f}" for j in range(len(kg))))
