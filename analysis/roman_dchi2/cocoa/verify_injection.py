#!/usr/bin/env python3
"""Verify the FLAMINGO -> cosmolike conversion and the injection.

1. Python conversion (fit_tatt_flamingo.load_snapshot):
       P_dE = (3/2) M0^{hE} / (2R) / b(k),   R = 1 - e_rms^2
       P_EE = (M0 - 5/2 M2 + 27/8 M4)^{EE} / (2R)^2 / <W^2_NGP>(k)
   recomputed here from the raw Pds files, term by term, and the low-k NLA
   amplitude compared with the 2026-10-27 report (aNLA, C1rho = 0.0134).
2. sim_IA.c doing the conversion itself: raw files (PdE0 = M0^{hE}/b, raw
   PEE0/2/4) injected with apply_shear = 1, los_mode = 3 (hybrid), NGP window
   on (681 Mpc/h, 1024), e_rms per snapshot, vs my pre-converted tables
   injected with no conversion. Same rows (nfold 1, 0.05 <= k <= 1.2), so
   the extrapolation is identical: the xi must agree.
3. Physical check: injected FLAMINGO xi_IA vs cosmolike's native NLA with
   the report's aNLA(z) (power law fitted to the table), large scales.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import fit_tatt_flamingo as F  # noqa: E402
from cosmolike_driver import Cosmolike  # noqa: E402
from cosmolike_xi_ia import write_set  # noqa: E402

# The raw FLAMINGO spectra (Pds files) ship in flamingo/data/PS of this
# repository; set FLAMINGO_PS to read them from elsewhere.
PS = os.environ.get("FLAMINGO_PS", os.path.join(HERE, "..", "..", "..", "flamingo", "data", "PS"))
REPORT = {78: 1.86, 76: 2.17, 74: 2.88, 72: 3.17, 70: 3.35, 68: 3.50, 66: 3.56, 63: 3.78,
          58: 4.14, 48: 4.94, 38: 6.04, 18: 8.32, 14: 10.02}


def main():
    data = sys.argv[1]
    out = open("analysis/verify_injection.txt", "w")

    def say(s):
        print(s, flush=True)
        out.write(s + "\n")

    model = F.TATTModel(halofit="takahashi")
    sns = sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0])
    zs = np.array([F.SNAPSHOTS[s][0] for s in sns])
    # ---- 1. term-by-term conversion and the report's aNLA ----
    say("1. conversion check (raw Pds files -> cosmolike convention), nfold 1")
    say("   sn    z   e_rms  1/(2R)   max|dE mine/recomputed - 1|  max|EE ...|  aNLA mine  report  ratio")
    for sn in sns:
        z, erms = F.SNAPSHOTS[sn][:2]
        s1 = 1/(2*(1 - erms**2))
        d = F.load_snapshot(PS, sn, 1)
        k, M0hE, _, _ = F._load(PS, "PhE0", sn, 1)
        EE = [F._load(PS, f"PEE{l}", sn, 1)[1] for l in (0, 2, 4)]
        W2 = F.ngp_w2_avg(k, F.BOX_HMPC/(1*F.NGRID))
        hE = 1.5*s1*M0hE
        ee = (EE[0] - 2.5*EE[1] + 27/8*EE[2])*s1**2/W2
        rdE = np.max(np.abs(d["hE"][0]/hE - 1))
        rEE = np.max(np.abs(d["EE"][0]/ee - 1))
        # low-k NLA amplitude, report recipe: P_dE = A P_NL at 0.05-0.3, mode-scatter weights
        PNL = model.Pnl(z, k)
        b = F.scale_bias(d, PNL)
        m = (k >= 0.05) & (k <= 0.3)
        T = -(F.COSMO["Om"]*F.C1RHOCRIT_REPORT/model.D(z))*b*PNL
        w = 1/d["hE"][1][m]**2
        a = float(np.sum(w*d["hE"][0][m]*T[m])/np.sum(w*T[m]**2))
        say(f"   {sn:3d} {z:4.2f}  {erms:.3f}  {s1:.4f}   {rdE:.1e}                      {rEE:.1e}      "
            f"{a:6.3f}   {REPORT[sn]:5.2f}  {a/REPORT[sn]:.3f}")

    # ---- 2. sim_IA.c conversion vs mine ----
    cl = Cosmolike(os.path.join(HERE, "lib"), data)
    h = cl.h
    xi_gg = cl.xi_native(ia_model=0, A1=[0.0, 0.0])
    rawdir = "run/tables_raw"
    os.makedirs(rawdir, exist_ok=True)
    mine_dE, mine_EE, karr = [], [], None
    for sn in sns:
        z, erms = F.SNAPSHOTS[sn][:2]
        d = F.load_snapshot(PS, sn, 1)
        k = d["k"]
        sel = (k >= 0.05) & (k <= 1.2)
        PNL = model.Pnl(z, k)
        b = F.scale_bias(d, PNL)
        kk, M0hE, _, _ = F._load(PS, "PhE0", sn, 1)
        np.savetxt(os.path.join(rawdir, f"PdE0_sn{sn}_nfold1.dat"),
                   np.column_stack([k[sel], (M0hE/b)[sel]]), fmt="%.8e")
        for l in (0, 2, 4):
            kl, Ml, _, _ = F._load(PS, f"PEE{l}", sn, 1)
            np.savetxt(os.path.join(rawdir, f"PEE{l}_sn{sn}_nfold1.dat"),
                       np.column_stack([kl[sel], Ml[sel]]), fmt="%.8e")
        mine_dE.append((d["hE"][0]/b)[sel])
        mine_EE.append(d["EE"][0][sel])
        karr = k[sel]
    cl.ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
    cl.ci.init_IA_sim(rawdir, [int(s) for s in sns], [float(z) for z in zs], 1)
    cl.ci.set_IA_sim_shape_convention(1, 3)
    cl.ci.set_IA_sim_window(1, 681.0, 1024)
    cl.ci.set_IA_sim_erms([int(s) for s in sns], [float(F.SNAPSHOTS[s][1]) for s in sns])
    cl.sim = True
    xi_simconv = cl.xi_sim() - xi_gg
    p = write_set("run/tables_mine", "mine", sns, np.array(mine_dE), np.array(mine_EE), karr)
    cl.use_sim_tables(p, sns, zs, 1)
    xi_mine = cl.xi_sim() - xi_gg
    r = np.abs(xi_simconv - xi_mine)/np.max(np.abs(xi_mine))
    say("\n2. sim_IA.c's own conversion (raw files, apply_shear 1, hybrid LOS, NGP window, e_rms) vs")
    say("   my pre-converted tables (same rows nfold 1, 0.05-1.2 h/Mpc), xi_IA over all 1080 points:")
    say(f"   max |difference| / max |xi_IA| = {r.max():.2e};  median ratio = "
        f"{np.median(xi_simconv[np.abs(xi_mine) > 0.05*np.abs(xi_mine).max()]/xi_mine[np.abs(xi_mine) > 0.05*np.abs(xi_mine).max()]):.6f}")

    # ---- 3. injected FLAMINGO vs native NLA with the report's aNLA(z) ----
    R = np.load("run/cosmolike_xi_ia.npz")
    a_cl = np.array([REPORT[s] for s in sns])*F.C1RHOCRIT_REPORT/F.C1RHOCRIT   # cosmolike units
    x = np.log((1 + zs)/1.62)
    eta, lnA = np.polyfit(x, np.log(a_cl), 1)
    nat = cl.xi_native(ia_model=0, A1=[float(np.exp(lnA)), float(eta)]) - xi_gg
    fl = R["xi_flam"]
    th = np.tile(np.arange(15), 72)
    say("\n3. injected FLAMINGO xi_IA vs cosmolike native NLA with the report's aNLA(z)")
    say(f"   (power law A1 = {np.exp(lnA):.3f}, eta = {eta:.3f}; cosmolike units)")
    for i0, lab in ((10, "theta >= 54'"), (7, "theta >= 21'"), (0, "all theta")):
        mm = (th >= i0) & (np.abs(nat) > 0.05*np.abs(nat).max())
        say(f"   {lab:13s}: median FLAMINGO / NLA = {np.median(fl[mm]/nat[mm]):.3f}  "
            f"(16-84%: {np.percentile(fl[mm]/nat[mm], 16):.3f} - {np.percentile(fl[mm]/nat[mm], 84):.3f})")
    out.close()


if __name__ == "__main__":
    main()
