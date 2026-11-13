"""Shared per-snapshot NLA / TATT fitter (as v3/v3_flamingo.py, production truth,
N_s fixed to the B plateau, three probes, k >= 0.05), returning the full
parameter covariance. Used by sim_strategies.py."""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import fit_tatt_flamingo as F  # noqa: E402

KMIN = 0.05
FREE = {"NLA": ["a1"], "TATT": ["a1", "a2", "bTA"]}
LO = {"a1": 0.0, "a2": -5.0, "bTA": 0.0}
HI = {"a1": 20.0, "a2": 5.0, "bTA": 2.0}
PROBES = ("hE", "EE", "BB")

f = np.load(os.path.join(HERE, "..", "v2", "fits.npz"))
v3 = np.load(os.path.join(HERE, "..", "v3", "v3.npz"))
zs, kd = f["zs"], f["k"][0]
y, err, T, b, Pnl, Ns = f["y"], f["err"], f["T"], f["b"], f["PNL"], f["Ns_true"]
kg, core = v3["kgrid"], v3["core"]
nsn = len(zs)
model = F.TATTModel(halofit="takahashi")
KER = [model.kernels(zs[s], kd) for s in range(nsn)]


def pred(s, a1, a2=0.0, bTA=0.0):
    pr = model.predict(zs[s], kd, a1, a2, bTA, 1.0, K=KER[s], PNL=Pnl[s], Ns=0.0, bk=b[s])
    return np.array([pr["hE"], pr["EE"] + Ns[s], pr["BB"] + Ns[s]])


def kmask(kmax):
    return (kd >= KMIN) & (kd <= kmax*(1 + 1e-9))


def fit(s, data, mdl, kmax, starts=None, sig=None):
    """Least-squares fit of model mdl to data[3, nk] at snapshot s over
    0.05 <= k <= kmax. Returns dict(x (a1, a2, bTA), chi2, dof, cov (free params))."""
    sig = err[s] if sig is None else sig
    m = kmask(kmax)
    names = FREE[mdl]

    def full(x):
        d = dict(a1=0.0, a2=0.0, bTA=0.0)
        d.update(zip(names, x))
        return d["a1"], d["a2"], d["bTA"]

    def resid(x):
        return ((data - pred(s, *full(x)))/sig)[:, m].ravel()
    if starts is None:
        starts = [[4.0]] if mdl == "NLA" else [[4.0, 0.0, 0.1], [4.0, -1.0, 0.5], [4.0, 1.0, 0.5], [4.0, 0.0, 1.0]]
    lo = np.array([LO[n] for n in names])
    hi = np.array([HI[n] for n in names])
    best = None
    for x0 in starts:
        r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    c2 = float(best.fun @ best.fun)
    cov = np.linalg.pinv(best.jac.T @ best.jac)
    return dict(x=np.array(full(best.x)), chi2=c2, dof=3*int(m.sum()) - len(names), cov=cov)


def cut_rule(ok):
    """Last k_max before two consecutive failures; nan if the first two fail."""
    for j in range(len(kg) - 1):
        if not ok[j] and not ok[j + 1]:
            return kg[j - 1] if j > 0 else np.nan
    return kg[-1] if ok[-1] else kg[-2]
