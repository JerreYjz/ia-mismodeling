#!/usr/bin/env python3
"""Roman figure of bias: HMc (central 1-halo, hmc_scan.py) against TATT and NLA
(same scan, same masks) and HM (hm_scan.py), cases A and B, K0/D(z) and
constant k_UV. -> fig_hmc_fob.png/pdf and a printed table."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
J = {f: json.load(open(os.path.join(HERE, f"hmc_scan_ex1_{f}.json"))) for f in ("D1", "flat")}
JH = {f: json.load(open(os.path.join(HERE, f"hm_scan_ex1_{f}.json"))) for f in ("D1", "flat")}
STY = {"NLA": ("#d1495b", ":", "NLA"), "TATT": ("#1f5fa8", "-", "TATT"), "HM": ("#9a9a9a", "--", "HM"),
       "HMc": ("#2a9d55", "-", "HMc (central 1-halo)")}

fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
for i, fam in enumerate(("D1", "flat")):
    rows = sorted(J[fam], key=lambda r: r["K0"])
    hrows = {r["K0"]: r for r in JH[fam]}
    for jc, case in enumerate(("B", "A")):
        a = ax[jc, i]
        for mdl in ("NLA", "TATT", "HM", "HMc"):
            src = [(r["K0"], hrows[r["K0"]]) for r in rows if r["K0"] in hrows] if mdl == "HM" else [(r["K0"], r) for r in rows]
            pts = [(K, r[f"{mdl}_{case}"]["FoB"], r[f"{mdl}_{case}"].get("FoB_sig") or 0) for K, r in src if f"{mdl}_{case}" in r]
            if not pts:
                continue
            K, F, S = map(np.array, zip(*pts))
            col, ls, lab = STY[mdl]
            a.plot(K, F, "o" + ls, color=col, lw=1.8 if mdl in ("TATT", "HMc") else 1.1, ms=4, label=lab)
            if mdl in ("TATT", "HMc"):
                a.fill_between(K, F - S, F + S, color=col, alpha=0.13, lw=0)
        a.axhline(0.3, color="k", lw=1, ls="--")
        a.set_title((r"$k_{\rm UV}(z) = K_0/D(z)$" if fam == "D1" else r"$k_{\rm UV} = K_0$") +
                    f", case {case}" + (r" ($\Omega_m$, $A_s$)" if case == "A" else r" (+ $n_s$, $H_0$, $\Omega_b$ priors)"), fontsize=10.5)
        a.set_xlabel(r"$K_0$ [$h\,$Mpc$^{-1}$]")
        a.grid(alpha=0.3)
        a.legend(fontsize=8.5, loc="upper left")
    ax[0, 0].set_ylabel(r"figure of bias in $(\Omega_m, S_8)$")
    ax[1, 0].set_ylabel(r"figure of bias in $(\Omega_m, S_8)$")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_hmc_fob.png"), dpi=220)
fig.savefig(os.path.join(HERE, "fig_hmc_fob.pdf"))

for fam in ("D1", "flat"):
    print(f"== {fam}:  K0 | case B: TATT FoB, HMc FoB (chi2 TATT, HMc) | case A: same")
    for r in sorted(J[fam], key=lambda r: r["K0"]):
        cells = []
        for case in ("B", "A"):
            t, c = r.get(f"TATT_{case}"), r.get(f"HMc_{case}")
            if t and c:
                cells.append(f"{t['FoB']:.2f}+-{t['FoB_sig']:.2f}  {c['FoB']:.2f}+-{c['FoB_sig']:.2f}  ({t['chi2']:.2f}, {c['chi2']:.2f})")
        print(f"  {r['K0']:5g} | " + " | ".join(cells))
