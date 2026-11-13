#!/usr/bin/env python3
"""Roman xi_+- Delta chi2 of NLA / TATT against the FLAMINGO-injected data
vector, on the production masks k_UV/D(z), three ways (all cosmolike):
  1. per-snapshot amplitudes fitted to FLAMINGO P(k) up to the FLAMINGO GOF
     cut (NLA k_max = 0.29, TATT k_max = 0.52; dchi2_masks.csv), injected
  2. power-law IA (cosmolike's own model) fitted to the Roman xi, cosmology
     fixed at the truth (bias_results.json, chi2_ia_only)
  3. power-law IA + Omega_m, A_s fitted (case A, chi2_bestfit)
Writes analysis/roman_dchi2_summary.csv and fig_roman_dchi2_summary."""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
KM = {"NLA": "0.291", "TATT": "0.523"}
rows = list(csv.DictReader(open("analysis/dchi2_masks.csv")))
B = json.load(open("analysis/bias_results.json"))
K = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
out = []
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
for mdl in COL:
    r = [x for x in rows if x["model"] == mdl and x["kmax"] == KM[mdl]][0]
    pk = [float(r[f"n1_{k:g}"]) for k in K]
    ia = [[b for b in B if b["data"] == "FLAMINGO" and b["case"] == "A" and b["model"] == mdl and b["mask"] == k][0] for k in K]
    iab = [[b for b in B if b["data"] == "FLAMINGO" and b["case"] == "B" and b["model"] == mdl and b["mask"] == k][0] for k in K]
    axs[0].plot(K, pk, "^:", color=COL[mdl], label=f"{mdl}: P(k)-fit amplitudes per snapshot (k_max {float(KM[mdl]):.2f})")
    axs[0].plot(K, [b["chi2_ia_only"] for b in ia], "s--", color=COL[mdl], label=f"{mdl}: IA fitted to Roman, cosmology fixed")
    axs[0].plot(K, [b["chi2_bestfit"] for b in ia], "o-", color=COL[mdl], lw=2, label=f"{mdl}: IA + Omega_m, A_s fitted")
    axs[1].plot(K, [b["FoB"] for b in ia], "o-", color=COL[mdl], lw=2, label=f"{mdl}: Omega_m, A_s free")
    axs[1].plot(K, [b["FoB"] for b in iab], "s--", color=COL[mdl], label=f"{mdl}: + n_s, H0, Omega_b free (priors)")
    for k, a, b1, b2, c in zip(K, pk, ia, iab, ia):
        out.append((mdl, k, b1["npts"], a, b1["chi2_ia_only"], b1["chi2_bestfit"], b1["FoB"], b2["FoB"],
                    b1["dOm"]/b1["sOm"], b1["dS8"]/b1["sS8"]))
axs[0].axhline(1, color="k", lw=1)
axs[0].set_yscale("log")
axs[0].set_ylabel(r"Roman $\Delta\chi^2$ (masked $\xi_\pm$, Gaussian cov)")
axs[0].set_title("model vs FLAMINGO-injected data vector")
axs[1].axhline(0.3, color="k", lw=1)
axs[1].set_ylabel("figure of bias in (Omega_m, S8)")
axs[1].set_title("parameter bias; line: 0.3 sigma criterion")
for ax in axs:
    ax.set_xlabel(r"IR/UV mask $k_{UV}/D(z)$, $k_{UV}\ [h\,{\rm Mpc}^{-1}]$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("analysis/fig_roman_dchi2_summary.png", dpi=220)
fig.savefig("analysis/fig_roman_dchi2_summary.pdf")
with open("analysis/roman_dchi2_summary.csv", "w") as fh:
    fh.write("model,mask_kUV_over_D,npts,dchi2_Pk_amplitudes,dchi2_IA_fit,dchi2_IA_cosmo_fit,FoB_A,FoB_B,dOm_over_sigma_A,dS8_over_sigma_A\n")
    for r in out:
        fh.write(",".join(str(x) if isinstance(x, str) else f"{x:.3g}" for x in r) + "\n")
print(open("analysis/roman_dchi2_summary.csv").read())
