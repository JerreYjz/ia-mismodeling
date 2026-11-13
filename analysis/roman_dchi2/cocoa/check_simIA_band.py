#!/usr/bin/env python3
"""What sim_IA's default trusted band does to a FLAMINGO injection.
sim_IA.c trims table rows to [SIM_KMIN_HMPC, SIM_KMAX_HMPC] (defaults
0.5, 7 h/Mpc) and extrapolates outside: power law in |P| below (log slope
floored at 0, i.e. flat at worst) and linear in P above. Emulated here by
writing only the rows inside a band (the library is built without trimming)
and compared with the main FLAMINGO tables (ratio to P_NL held outside the
measured 0.05 - 17.7 h/Mpc). Delta chi2 on all points and on the IR/UV
masks k_UV/D(z)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cosmolike_driver import Cosmolike
from cosmolike_xi_ia import write_set
d = np.load("../v2/fits.npz"); R = np.load("run/cosmolike_xi_ia.npz")
cov = np.load("analysis/cov_gauss_d3a.npy")
zs, sns, kd = d["zs"], d["sns"], d["k"][0]
cl = Cosmolike(os.path.join(HERE, "lib"), sys.argv[1])
h = cl.h
T, b, Pf, Ns = d["T"], d["b"], d["PNL"], d["Ns_true"]
xi_gg = R["xi_gg"]
q = lambda v, m: float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))
M1, K1 = R["masks_n1"], R["K_n1"]
allm = np.ones(1080, bool)
out = {}
for lo, hi in ((0.05, 17.7), (0.3, 7.0), (0.5, 7.0)):
    m = (kd >= lo*0.999) & (kd <= hi*1.001)
    PdE = np.array([(T[s, 0]/(b[s]*Pf[s]))[m]*cl.PKNL.P(zs[s], kd[m]*h)*h**3 for s in range(len(zs))])
    PEE = np.array([((T[s, 1] - Ns[s])/Pf[s])[m]*cl.PKNL.P(zs[s], kd[m]*h)*h**3 for s in range(len(zs))])
    p = write_set("run/tables_band", f"band_{lo}_{hi}", sns, PdE, PEE, kd[m])
    cl.use_sim_tables(p, sns, zs, 1)
    dv = cl.xi_sim() - xi_gg - R["xi_flam"]
    row = [q(dv, allm)] + [q(dv, mm) for mm in M1]
    out[(lo, hi)] = row
    print(f"rows {lo}-{hi} h/Mpc: Delta chi2 vs main FLAMINGO: all {row[0]:9.1f}; " +
          ", ".join(f"{k:g}/D {v:7.2f}" for k, v in zip(K1, row[1:])), flush=True)
np.save("analysis/simIA_band_check.npy", out, allow_pickle=True)
