"""Planck parameter covariance for gn_bias.py --ext-cov (Option A).

Reads a Planck Legacy Archive (or any getdist / Cobaya) chain and writes the
weighted covariance of the cosmological parameters gn_bias.py varies:

    As_1e9  ns  H0  omegab  omegam

in getdist .covmat format ('# names' line, then the matrix). Only the
covariance is used: gn_bias.py centres the prior on the Roman truth, so the
chain's own best fit does not matter. tau and the Planck nuisances are
marginalised automatically (they are simply left out).

  python make_ext_cov.py \\
      --root /path/to/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE \\
      --out planck18_TTTEEE_lowl_lowE_cov.txt

--root is the chain root: the files root_1.txt, root_2.txt, ... and
root.paramnames. --ignore-rows is the burn-in fraction (PLA chains: 0.3).
Add --with-lensing-root for a chain that also includes CMB lensing, e.g.
base_plikHM_TTTEEE_lowl_lowE_lensing, if you want that variant.
"""
import argparse

import numpy as np
from getdist import loadMCSamples

OUT = ["As_1e9", "ns", "H0", "omegab", "omegam"]


def column(p, *names):
    for n in names:
        v = getattr(p, n, None)
        if v is not None:
            return np.asarray(v)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ignore-rows", type=float, default=0.3)
    a = ap.parse_args()

    s = loadMCSamples(a.root, settings={"ignore_rows": a.ignore_rows})
    p = s.getParams()
    H0 = column(p, "H0")
    As = column(p, "A", "As_1e9")                 # PLA: A = 10^9 A_s
    if As is None:
        logA = column(p, "logA", "logA_1e10")     # ln(10^10 A_s)
        As = np.exp(logA)/10.0
    ns = column(p, "ns")
    ombh2 = column(p, "omegabh2", "ombh2")
    omb = column(p, "omegab")
    if omb is None:
        omb = ombh2/(H0/100.0)**2
    om = column(p, "omegam", "omegam_")
    cols = dict(As_1e9=As, ns=ns, H0=H0, omegab=omb, omegam=om)
    miss = [k for k, v in cols.items() if v is None]
    if miss:
        raise SystemExit(f"chain lacks {miss}; its parameters are "
                         f"{[q.name for q in s.paramNames.names]}")

    X = np.column_stack([cols[k] for k in OUT])
    w = np.asarray(s.weights, dtype=float)
    mean = np.average(X, axis=0, weights=w)
    C = np.cov(X, rowvar=False, aweights=w)
    np.savetxt(a.out, C, header=" ".join(OUT), comments="# ", fmt="%.8e")
    sig = np.sqrt(np.diag(C))
    print(f"{s.numrows} samples (after {a.ignore_rows:g} burn-in)")
    for k, m, e in zip(OUT, mean, sig):
        print(f"  {k:7s} mean {m:.5f}  sigma {e:.5f}  ({100*e/abs(m):.2f}%)")
    R = C/np.outer(sig, sig)
    print("correlations:\n", np.array2string(R, precision=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
