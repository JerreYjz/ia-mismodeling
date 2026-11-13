#!/usr/bin/env python3
"""Validate the Cobaya-free cosmolike driver against cocoa_roman_real's
example1.modelvector (EXAMPLE_EVALUATE1: As 2.1e-9, ns 0.96605, H0 67.32,
Omega_b 0.04, Omega_m 0.3, mnu 0.06, NLA A1 = 0.6, eta = -1.5, lmax 75000),
and the python Limber code (roman_limber.py) against cosmolike."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cosmolike_driver import Cosmolike
DATA = sys.argv[1]
ev1 = dict(As=2.1e-9, ns=0.96605, H0=67.32, omegab=0.04, omegam=0.3, mnu=0.06, tau=0.0697186,
           w=-1.0, wa=0.0)
t = time.time()
cl = Cosmolike(os.path.join(HERE, "lib"), DATA, cosmo=ev1, lmax=75000, camb_boost=1.05, camb_kmax=7.5)
print(f"setup {time.time() - t:.1f}s, sigma8 = {cl.sigma8:.4f}", flush=True)
t = time.time()
xi = cl.xi_native(ia_model=0, A1=[0.6, -1.5])
print(f"xi: {xi.shape} in {time.time() - t:.1f}s", flush=True)
mv = np.loadtxt(os.path.join(DATA, "example1.modelvector"))[:, 1]
n = len(xi)
r = xi/mv[:n]
print("percentiles 1, 50, 99:", np.percentile(r, [1, 50, 99]))
print("ratio cosmolike-driver / example1.modelvector: median %.5f, min %.5f, max %.5f"
      % (np.median(r), r.min(), r.max()))
np.save(os.path.join(HERE, "val_xi_ev1.npy"), xi)
