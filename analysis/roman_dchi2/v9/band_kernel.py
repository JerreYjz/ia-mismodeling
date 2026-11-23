#!/usr/bin/env python3
"""Why the v9 GI band has two narrow peaks: the band with the IA amplitude fixed or free.

The band of ia_budget.py (section band) is the rms of a random error of one
spectrum, confined to a Gaussian window of width 0.3 in ln k around k_c
(exponential kernel, ell = 1, the same error at every redshift), at which the
rms shift of Omega_m or S8 reaches 0.3 sigma; production truth, no cut. It is
about 0.3 sigma over the shift that a one-sign bump at k_c causes. This
script recomputes it, and that shift, for three fits:
    fixed   Omega_m, A_s free; the template A, eta fixed
    A       Omega_m, A_s, A free; eta fixed
    A_eta   Omega_m, A_s, A, eta free (the band of the note)
"Fixed" is a prior precision of 1e12 on the parameter. The bump is
delta = w(k) at every node and snapshot (100%, one sign), w the window.
Checks printed: the A_eta band against ia_budget.json, and the shift of the
bump at k_c = 0.32 against a direct least-squares fit in data space.

Output: notes/ia_error_budget/plots/band_kernel/data.json.

    python band_kernel.py        (cocoa Python)
"""
import json
import os
import sys

import numpy as np
import scipy.linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ia_fisher as IF                                             # noqa: E402

OUT = os.path.join(HERE, "..", "..", "..", "notes", "ia_error_budget", "plots", "band_kernel")
FITS = {"fixed": (0, 1), "A": (0, 1, 2), "A_eta": (0, 1, 2, 3)}
WK, ELL = 0.3, 1.0


def main():
    base = IF.Base(os.path.join(HERE, "ia_base_prod.npz"), os.path.join(HERE, "ia_base_common.npz"))
    zs, k = base.zs, IF.NODES
    nz = len(zs)
    mask = base.masks["none"]
    G = IF.gram(base.B, base.cov, mask)
    out = dict(about=__doc__.split("\n\n")[0] + " Producer: analysis/roman_dchi2/v9/band_kernel.py; "
               "band = 0.3/max_p rms shift per unit rms error (fraction), shift = Omega_m and S8 shift in "
               "sigma of a 100% one-sign bump at k_c. Production truth, no cut, ell = 1, L_z = inf.",
               k_c=k.tolist(), fits={})
    for X in ("GI", "II"):
        Rm = base.knob_map((X,))
        K = IF.residual_cov(k, zs, ELL, np.inf, "exp", (X,), 0.0)
        for name, free in FITS.items():
            M = np.zeros((base.nb, base.npar))
            M[base.ij, base.ij] = 1.0
            P = IF.template_prior(base.npar)
            for j in range(base.npar):
                if j not in free:
                    P[j, j] = 1e12
            fit = IF.Fit(G, M, P, P, base.g.copy())
            A = fit.shift(Rm)
            band, shift = [], []
            for kc in k:
                w = np.exp(-np.log(k/kc)**2/(2*WK**2))
                Aw = A*np.tile(w, nz)[None, :]
                s = np.sqrt(np.einsum("pi,ij,pj->p", Aw, K, Aw))
                band.append(float(min(IF.TOL/s.max(), IF.SIGMA_CAP)))   # capped as in rms_budget
                shift.append((A.reshape(2, nz, -1).sum(1)*w[None, :]).sum(1).tolist())
            out["fits"][f"{X}|{name}"] = dict(band=band, shift=shift, sigma=fit.sig.tolist())
            sh = np.array(shift)[:, 0]
            cross = [float(np.sqrt(k[i]*k[i + 1])) for i in range(len(k) - 1) if sh[i]*sh[i + 1] < 0]
            print(f"{X} {name:6s}: sigma(Om, S8) {np.round(fit.sig, 4).tolist()}, Omega_m shift crosses 0 at "
                  f"{np.round(cross, 2).tolist()}, band max at k <= 3: "
                  f"{max(b for b, kk in zip(band, k) if kk <= 3.2):.3f}")
    # check 1: the A_eta band against ia_budget.json
    ref = json.load(open(os.path.join(HERE, "ia_budget.json")))
    for X in ("GI", "II"):
        r = np.array(ref["truths"]["prod"]["none"]["band"][f"{X}|1|inf"]["sigma_max"])
        b = np.array(out["fits"][f"{X}|A_eta"]["band"])
        print(f"check: {X} band against ia_budget.json, largest relative difference {np.abs(b/r - 1).max():.1e}")
    # check 2: direct least squares for the GI bump at k_c = 0.32
    Rm = base.knob_map(("GI",))
    w = np.exp(-np.log(k/0.316)**2/(2*WK**2))
    dd = (base.B @ Rm)[mask] @ np.tile(w, nz)
    L = scipy.linalg.cholesky(base.cov[np.ix_(mask, mask)], lower=True)
    J = base.B[mask][:, :base.npar]
    P = IF.template_prior(base.npar)
    Aw = np.vstack([scipy.linalg.solve_triangular(L, J, lower=True), np.diag(np.sqrt(np.diag(P)))])
    bw = np.concatenate([scipy.linalg.solve_triangular(L, dd, lower=True), np.zeros(base.npar)])
    p = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    sig = np.array(out["fits"]["GI|A_eta"]["sigma"])
    i = int(np.argmin(np.abs(k - 0.316)))
    print(f"check: bump at 0.32, direct lstsq {np.round(base.g @ p/sig, 3).tolist()} against "
          f"{np.round(out['fits']['GI|A_eta']['shift'][i], 3).tolist()}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
