#!/usr/bin/env python3
"""Where does the Roman real-space 'leakage' come from? For a model and fit
k_max, split the residual dP (white offset removed) by spectrum (dE, EE, BB)
and by k range (0.05..K matched, K..17.7 measured, beyond 17.7 extrapolated),
and report Delta chi2_R on the IR/UV mask at K = k_max (real and Fourier)."""
import sys
import numpy as np
sys.path.insert(0, ".."); sys.path.insert(0, ".")
import roman_limber as RL
import roman_map as M

nzr, nzf = sys.argv[1], sys.argv[2]
d = np.load("fits.npz"); zs = d["zs"]; k = d["k"][0]; kg = d["kgrid"]; nsn = len(zs)
norm = np.array([[d["b"][i]*d["PNL"][i], d["PNL"][i], d["PNL"][i]] for i in range(nsn)])
c = RL.Cosmology(**M.FLAMINGO)
sp = {"real": RL.RomanShear(c, nzr), "fourier": RL.RomanFourier(c, nzf)}
T = M.ratios(zs, k, d["Rtrue"])
cov, masks, vec = {}, {}, {}
for n, S in sp.items():
    CEg, CBg = S.cl(None); CEi, CBi = S.cl(T, gg=False)
    if n == "real":
        cov[n] = S.gaussian_cov(CEg + CEi, CBg + CBi); vec[n] = lambda CE, CB, S=S: S.xi(CE, CB)
    else:
        cov[n] = S.gaussian_cov(CEg + CEi); vec[n] = lambda CE, CB, S=S: S.bandpowers(CE)
    masks[n] = np.load(f"roman/masks_{n}.npy")


def tables(dP, cw, klo, khi, comps, extrap):
    lo = np.geomspace(1e-4, M.KMIN, 25, endpoint=False); hi = np.geomspace(k[-1], 2e3, 26)[1:]
    m = k >= M.KMIN; kk = np.concatenate([lo, k[m], hi])
    Rt = np.zeros((nsn, 3, len(kk)))
    for s, z in enumerate(zs):
        pnl = c.Pnl.P(z, kk)
        keep = ((k[m] >= klo) & (k[m] <= khi*(1 + 1e-9))).astype(float)
        e = float(extrap)
        r = dP[s, 0, m]/norm[s, 0, m]
        if 0 in comps:
            Rt[s, 0] = np.concatenate([np.zeros(len(lo)), r*keep, np.full(len(hi), r[-1]*e)])
        for x in (1, 2):
            if x in comps:
                p = dP[s, x, m] - cw[s]
                Rt[s, x] = np.concatenate([np.zeros(len(lo)), p*keep, np.full(len(hi), p[-1]*e)])/pnl
    return {x: RL.ratio_table(zs, kk, Rt[:, i, :]) for i, x in enumerate(M.X)}


for mdl, j in (("NLA", 1), ("NLA", 4), ("TATT", 1), ("TATT", 6)):
    K = kg[j]
    x = d[f"{mdl}_x_true"][:, j]; Ns = x[:, -1]
    Mobs = d[f"{mdl}_R"][:, j]*norm; Mobs[:, 1] += Ns[:, None]; Mobs[:, 2] += Ns[:, None]
    dP = Mobs - d["T"]
    cw = [M.dchi2_white(dP[s], d["err"][s], k, K)[1] for s in range(nsn)]
    print(f"\n{mdl} k_max = K = {K:.3f}")
    for lab, klo, khi, ex in (("0.05..K (matched)", 0.05, K, False), ("K..17.7 (measured)", K*1.0001, 99, False),
                              (">17.7 (extrapolated)", 99, 99, True), ("all", 0.05, 99, True)):
        row = []
        for comps, cl in (((0,), "dE"), ((1,), "EE"), ((2,), "BB"), ((0, 1, 2), "all")):
            for n in sp:
                CE, CB = sp[n].cl(tables(dP, cw, klo, khi, comps, ex), gg=False)
                row.append(f"{cl}/{n[0]} {M.qform(vec[n](CE, CB), cov[n], masks[n][j]):8.3f}")
        print(f"  {lab:22s} " + "  ".join(row))
