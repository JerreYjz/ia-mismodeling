#!/usr/bin/env python3
"""fig_roman_test: FoB and chi2 vs cut, cases A and B, with the controls; roman_test.csv."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
res = json.load(open("roman_test.json"))
Mz = np.load("masks_v3.npz")
z2 = Mz["z2"]
kuv_flat = {f: float(np.exp(np.interp(1.0, z2, np.log(f*Mz["kuv_TATT_GoF"])))) for f in (0.7, 1.0, 1.4, 2.0)}


def rows(fam, case, test):
    r = [x for x in res if x["family"] == fam and x["case"] == case and x["test"] == test]
    if fam == "K0/D":
        r.sort(key=lambda x: x["K0"])
        xs = [x["K0"] for x in r]
    else:
        r.sort(key=lambda x: x["factor"])
        xs = [kuv_flat[x["factor"]] for x in r]
    return np.array(xs), r


fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for col, fam, xl in ((0, "K0/D", r"mask $K_0$: $k_{\rm UV}=K_0/D(z)$ [h/Mpc]"), (1, "flat", r"flat cut: $k_{\rm UV}(z=1)$ [h/Mpc]")):
    for row, case in ((0, "A"), (1, "B")):
        a = ax[row, col]
        for mdl in ("NLA", "TATT"):
            x, r = rows(fam, case, f"{mdl}_on_FLAMINGO")
            fob = np.array([v["FoB"] for v in r])
            sig = np.array([v.get("FoB_sig", 0) for v in r])
            a.plot(x, fob, "-o", ms=5, color=COL[mdl], label=f"{mdl} on FLAMINGO")
            a.fill_between(x, fob - sig, fob + sig, color=COL[mdl], alpha=0.12, lw=0)
        x, r = rows(fam, case, "TATT_on_TATT")
        a.plot(x, [v["FoB"] for v in r], ":^", ms=5, color=COL["TATT"], label="control: TATT on TATT truth")
        x, r = rows(fam, case, "NLA_on_TATT")
        a.plot(x, [v["FoB"] for v in r], "--s", ms=5, color="0.35", label="control: NLA on TATT truth (Part 1)")
        x, r = rows(fam, case, "NLA_on_NLA_injected")
        a.plot(x, [v["FoB"] for v in r], ":v", ms=5, color=COL["NLA"], label="control: NLA on NLA truth via sim tables")
        a.axhline(0.3, color="k", lw=0.8, ls="--")
        a.set_ylabel(r"FoB in $(\Omega_m, S_8)$ [$\sigma$]")
        a.set_title(f"case {case}: " + (r"$\Omega_m, A_s$ free" if case == "A" else r"$\Omega_m, A_s, n_s, H_0, \Omega_b$ free") + f", {fam} masks")
        a.set_xlabel(xl)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
        if fam == "flat":
            a.set_xscale("log")
            a.set_xticks([0.2, 0.3, 0.4, 0.6])
            a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            a.xaxis.set_minor_formatter(ticker.NullFormatter())
    for row, case in ((0, "A"), (1, "B")):
        a = ax[row, 2]
        for mdl in ("NLA", "TATT"):
            x, r = rows(fam, case, f"{mdl}_on_FLAMINGO")
            a.plot(x, [v["chi2_bestfit"] for v in r], "-o" if fam == "K0/D" else "--s", ms=5, color=COL[mdl],
                   label=f"{mdl}, {fam} masks" + (" (x: $K_0$)" if fam == "K0/D" else " (x: $k_{\\rm UV}(z=1)$)"))
        a.axhline(1, color="k", lw=0.8, ls="--")
        a.set_ylabel(r"$\chi^2$ at the best fit")
        a.set_title(f"case {case}: residual misfit (FLAMINGO truth)")
        a.set_xlabel("cut [h/Mpc]")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
fig.tight_layout()
fig.savefig("fig_roman_test.png", dpi=220)
fig.savefig("fig_roman_test.pdf")

with open("roman_test.csv", "w") as fh:
    fh.write("family,mask,case,test,npts,chi2,dOm_sig,dS8_sig,FoB,FoB_sig,at_bound\n")
    for v in res:
        fh.write(f"{v['family']},{v['mask']},{v['case']},{v['test']},{v['npts']},{v['chi2_bestfit']:.3f},"
                 f"{v['dOm']/v['sOm']:+.2f},{v['dS8']/v['sS8']:+.2f},{v['FoB']:.3f},{v.get('FoB_sig', float('nan')):.3f},{'|'.join(v['at_bound'])}\n")
print(open("roman_test.csv").read())
