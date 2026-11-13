#!/usr/bin/env python3
"""fig_mask_scan (three mask families) and fig_ggl3x2 (GGL / 3x2pt); CSV tables."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
FAM = {"D1": (r"$K_0/D(z)$  (n = $-$1)", "-"), "D23": (r"$K_0/D(z)^{2/3}$  (n = 0)", "--"), "flat": (r"$K_0$ constant", ":")}

# ---------------- mask scan ----------------
if os.path.exists("mask_scan.json"):
    res = json.load(open("mask_scan.json"))
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))
    for row, mdl in enumerate(("NLA", "TATT")):
        for fam, (lab, ls) in FAM.items():
            r = sorted([x for x in res if x["family"] == fam], key=lambda x: x["K0"])
            K = [x["K0"] for x in r]
            a = ax[row, 0]
            fob = np.array([x[f"{mdl}_A"]["FoB"] for x in r])
            sig = np.array([x[f"{mdl}_A"]["FoB_sig"] or 0 for x in r])
            a.plot(K, fob, ls, marker="o", ms=4, color=COL[mdl], label=f"{lab}, case A")
            a.fill_between(K, fob - sig, fob + sig, color=COL[mdl], alpha=0.1, lw=0)
            a.plot(K, [x[f"{mdl}_B"]["FoB"] for x in r], ls, marker="^", ms=4, mfc="none", color="0.3", label=f"{lab}, case B")
            a = ax[row, 1]
            a.plot(K, fob, ls, marker="o", ms=4, color=COL[mdl], label=f"{lab}: {mdl}")
            a.plot(K, [x[f"{mdl}w_A"]["FoB"] for x in r], ls, marker="x", ms=7, mew=1.8, color="#2a9d8f", label=f"{lab}: {mdl} + white template")
            a = ax[row, 2]
            a.plot(K, [x[f"{mdl}_A"]["chi2"] for x in r], ls, marker="o", ms=4, color=COL[mdl], label=f"{lab}: {mdl}")
            a.plot(K, [x[f"{mdl}w_A"]["chi2"] for x in r], ls, marker="x", ms=7, mew=1.8, color="#2a9d8f", label=f"{lab}: {mdl} + white template")
            for x in r:
                ax[row, 0].annotate(f"{x['npts']}", (x["K0"], x[f"{mdl}_A"]["FoB"]), textcoords="offset points", xytext=(0, -11), fontsize=6, ha="center", color="0.4")
        ax[row, 0].set_title(f"{mdl} fitted: FoB, three mask families (numbers: points kept)")
        ax[row, 1].set_title(f"{mdl} fitted: with and without the white II template (case A)")
        ax[row, 2].set_title(f"{mdl} fitted: $\\chi^2$ at the best fit (case A)")
        for c in range(3):
            a = ax[row, c]
            a.axhline(0.3 if c < 2 else 1.0, color="k", lw=0.8, ls="--")
            a.set_xlabel(r"$K_0$ [h/Mpc]")
            a.set_ylabel(r"FoB in $(\Omega_m, S_8)$ [$\sigma$]" if c < 2 else r"$\chi^2$ at the best fit")
            a.grid(alpha=0.3)
            a.legend(fontsize=6.5)
            if c < 2:
                a.set_ylim(0, 0.75)
    fig.tight_layout()
    fig.savefig("fig_mask_scan.png", dpi=220)
    fig.savefig("fig_mask_scan.pdf")
    with open("mask_scan.csv", "w") as fh:
        fh.write("family,K0,kuv_z0.5,kuv_z1,kuv_z2,npts,NLA_A_FoB,NLA_A_sig,NLA_A_chi2,NLA_A_dOm,NLA_A_dS8,TATT_A_FoB,TATT_A_sig,TATT_A_chi2,TATT_A_dOm,TATT_A_dS8,NLA_B_FoB,TATT_B_FoB,NLAw_FoB,NLAw_chi2,TATTw_FoB,TATTw_chi2,TATT_A_bound\n")
        for x in sorted(res, key=lambda x: (x["family"], x["K0"])):
            n, t = x["NLA_A"], x["TATT_A"]
            fh.write(f"{x['family']},{x['K0']},{x['kuv_z05']:.3f},{x['kuv_z1']:.3f},{x['kuv_z2']:.3f},{x['npts']},"
                     f"{n['FoB']:.3f},{n['FoB_sig'] or 0:.3f},{n['chi2']:.2f},{n['dOm']:+.2f},{n['dS8']:+.2f},"
                     f"{t['FoB']:.3f},{t['FoB_sig'] or 0:.3f},{t['chi2']:.2f},{t['dOm']:+.2f},{t['dS8']:+.2f},"
                     f"{x['NLA_B']['FoB']:.3f},{x['TATT_B']['FoB']:.3f},{x['NLAw_A']['FoB']:.3f},{x['NLAw_A']['chi2']:.2f},"
                     f"{x['TATTw_A']['FoB']:.3f},{x['TATTw_A']['chi2']:.2f},{'|'.join(t['at_bound'])}\n")
    print(open("mask_scan.csv").read())

# ---------------- GGL / 3x2pt ----------------
f3 = "ggl3x2_all.json"
if os.path.exists(f3):
    res = json.load(open(f3))
    for x in res:
        x["cfg"] = "all pairs"
    for fb in ("ggl3x2_3x2_behind.json", "ggl3x2_ggl_behind.json"):
        if os.path.exists(fb):
            for x in json.load(open(fb)):
                x["cfg"] = "behind"
                res.append(x)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for col, cfg in enumerate(("all pairs", "behind")):
        for probe, cs, ls, mk in (("GGL+w", None, ":", "s"), ("3x2pt", 0.2, "-", "o"), ("3x2pt", 0.3, "--", "^")):
            r = sorted([x for x in res if x["probe"] == probe and x["K0_cs"] == cs and x["cfg"] == cfg], key=lambda x: x["K0_ggl"])
            if not r:
                continue
            K = [x["K0_ggl"] for x in r]
            lab = "GGL + w only" if probe == "GGL+w" else f"3x2pt, cosmic shear $K_0$ = {cs}"
            for mdl in ("NLA", "TATT"):
                ax[0, col].plot(K, [x[mdl]["FoB"] for x in r], ls, marker=mk, ms=5, color=COL[mdl], label=f"{mdl}: {lab}")
                ax[1, col].plot(K, [x[mdl]["chi2"] for x in r], ls, marker=mk, ms=5, color=COL[mdl], label=f"{mdl}: {lab}")
            for x in r:
                ax[0, col].annotate(f"{x['npts']['gt']}", (x["K0_ggl"], x["NLA"]["FoB"]), textcoords="offset points", xytext=(0, -11), fontsize=6, ha="center", color="0.4")
        ttl = "all 61 lens-source pairs" if cfg == "all pairs" else "sources behind lenses only (s > l, 28 pairs)"
        ax[0, col].set_title(f"FoB vs the GGL cut, {ttl}\n(numbers: $\\gamma_t$ points kept)", fontsize=10)
        ax[1, col].set_title(f"$\\chi^2$ at the best fit, {ttl}", fontsize=10)
        ax[0, col].axhline(0.3, color="k", lw=0.8, ls="--")
        ax[0, col].set_ylim(0, 1.6 if cfg == "all pairs" else 0.6)
        ax[1, col].set_yscale("log")
        ax[0, col].set_ylabel(r"FoB in $(\Omega_m, S_8)$ [$\sigma$]")
        ax[1, col].set_ylabel(r"$\chi^2$ at the best fit")
        for a in ax[:, col]:
            a.set_xlabel(r"$K_0^{\rm ggl}$ [h/Mpc]  ($k_{\rm UV}=K_0/D(z)$; clustering $K_0^{\rm gc}$ = 0.2)")
            a.grid(alpha=0.3)
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_ggl3x2.png", dpi=220)
    fig.savefig("fig_ggl3x2.pdf")
    with open("ggl3x2.csv", "w") as fh:
        fh.write("cfg,probe,K0_cs,K0_ggl,K0_gc,n_xi,n_gt,n_w,NLA_FoB,NLA_chi2,NLA_dOm,NLA_dS8,TATT_FoB,TATT_chi2,TATT_dOm,TATT_dS8,TATT_bound,NLA_sOm,NLA_sS8\n")
        for x in res:
            n, t = x["NLA"], x["TATT"]
            fh.write(f"{x['cfg']},{x['probe']},{x['K0_cs']},{x['K0_ggl']},{x['K0_gc']},{x['npts']['xi']},{x['npts']['gt']},{x['npts']['w']},"
                     f"{n['FoB']:.3f},{n['chi2']:.2f},{n['dOm']:+.2f},{n['dS8']:+.2f},{t['FoB']:.3f},{t['chi2']:.2f},{t['dOm']:+.2f},{t['dS8']:+.2f},{'|'.join(t['at_bound'])},{n['sOm']:.4f},{n['sS8']:.4f}\n")
    print(open("ggl3x2.csv").read())
