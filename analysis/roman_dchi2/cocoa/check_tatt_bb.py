#!/usr/bin/env python3
"""Is the native-TATT vs TATT-tables difference in cosmolike the IA B modes
(which the sim_IA path does not carry)? Python Limber: xi from the TATT
power-law model's P_BB alone, compared with cosmolike (native - tables)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import roman_limber as RL
import fit_tatt_flamingo as F
R = np.load("run/cosmolike_xi_ia.npz"); cov = np.load("analysis/cov_gauss_d3a.npy")
zs = R["zs"]; x = (1 + zs)/1.62
model = F.TATTModel(halofit="takahashi")
k = np.geomspace(1.001e-4, 25, 300)
RBB = []
for s, z in enumerate(zs):
    pr = model.predict(z, k, 3.8*x[s]**0.57, -1.08*x[s]**-0.71, 0.18, 1.0, Ns=0.0)
    RBB.append(pr["BB"]/model.Pnl(z, k))
RBB = np.array(RBB)
cosmo = RL.Cosmology(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, As=2.099e-9, mnu=0.06)
S = RL.RomanShear(cosmo, sys.argv[1])
zero = lambda K, Z: 0*K
CE, CB = S.cl({"dE": zero, "EE": zero, "BB": RL.ratio_table(zs, k, RBB)}, gg=False)
xbb = S.xi(CE, CB)
diff = R["check_TATT_native"] - R["check_TATT_tables"]
q = lambda v: float(v @ np.linalg.solve(cov, v))
print(f"cosmolike native - tables: chi2 {q(diff):.2f};  python TATT B-mode xi: chi2 {q(xbb):.2f};  "
      f"residual (native - tables - BB): chi2 {q(diff - xbb):.2f}")
c = np.corrcoef(diff, xbb)[0, 1]
print(f"correlation of the two difference vectors: {c:.3f}")
