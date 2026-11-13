#!/usr/bin/env python3
"""FLAMINGO 'truth' and per-snapshot NLA / TATT IA spectra for the Limber step.

Truth, per snapshot (FLAMINGO notes' stitch):
    P_X^true = [1 - w(k)] P_X^NLA + w(k) P_X^sim,     X = dE, EE, BB
    w = logistic in ln k at k_t(z) (KT of fit_stitched_kmax.py), width 0.30
    P_dE^sim = P_hE / b(k), b = sqrt(P_hh / P_NL)       (as the fits)
    P_EE^sim, P_BB^sim: measured minus the white residual Ns
    NLA side: a1 and Ns fitted at 0.05 <= k <= 0.3 (fit_stitched_kmax.nla_fit)
--truth bconst (in roman_dchi2.py) uses P_hE / b0 with b0 the large-scale
bias (mean b(k) at 0.05-0.2) instead of P_hE / b(k): systematics check on
the deterministic-bias convention at k > 1. truth_err holds the FLAMINGO
1-sigma errors of the stitched ratios (sim side, x w) for a noise floor.
Folds: nfold 1 for k < 1.2, nfold 4 for 1.2 <= k < 9, nfold 16 above (to
k ~ 17.7 h/Mpc). --ns-mode highk takes Ns from the median P_BB of the nfold 4
data at 5 < k < 10 instead (systematics check on the BB/EE floor).

Models, per snapshot and per k_max on the data k grid: amplitudes only
(NLA a1; TATT a1, a2, b_TA; + Ns for P_EE), fitted to P_hE + P_EE over
0.05 <= k <= k_max with FLAMINGO's measured error bars, no floor
(gof_per_snapshot.fit, --err mode). Model spectra from TATTModel.predict
with Ns = 0.

Everything is stored as R_X = P_X / P_NL on the common k grid (CAMB P_NL,
FLAMINGO cosmology, takahashi), together with the P(k)-level chi2:
    chi2_fit(k_max)  : the fit's own chi2 over k <= k_max (P_hE + P_EE)
    chi2_wide(k_max) : the same model against the data over 0.05 <= k <= --kwide

    python build_spectra.py --ps-dir PS --out spectra.npz
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tatt_fit"))
import fit_tatt_flamingo as F  # noqa: E402
import gof_per_snapshot as G   # noqa: E402
from fit_stitched_kmax import KT, SIGMA_W, weight, nla_fit  # noqa: E402

FOLDS = [(1, 0.0, 1.2), (4, 1.2, 9.0), (16, 9.0, 1e9)]


def combined(ps, sn, model):
    """Spectra on the fold-combined k grid: k, P_NL, b(k), hE, EE, BB."""
    ks, out = [], {p: [] for p in ("hh", "hE", "EE", "BB")}
    err = {p: [] for p in ("hE", "EE", "BB")}
    for nf, lo, hi in FOLDS:
        d = F.load_snapshot(ps, sn, nf)
        m = (d["k"] >= lo) & (d["k"] < hi)
        ks.append(d["k"][m])
        for p in out:
            out[p].append(d[p][0][m])
        for p in err:
            err[p].append(d[p][1][m])
    k = np.concatenate(ks)
    o = np.argsort(k)
    k = k[o]
    sp = {p: np.concatenate(v)[o] for p, v in out.items()}
    sp.update({"s" + p: np.concatenate(v)[o] for p, v in err.items()})
    z = F.SNAPSHOTS[sn][0]
    PNL = model.Pnl(z, k)
    b = F.scale_bias({"hh": (sp["hh"], None)}, PNL)
    return k, PNL, b, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", default="spectra.npz")
    ap.add_argument("--kwide", type=float, default=1.7)
    ap.add_argument("--halofit", default="takahashi")
    a = ap.parse_args()

    print("CAMB + FAST-PT ...", flush=True)
    model = F.TATTModel(halofit=a.halofit)
    sns = sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0])
    zs = np.array([F.SNAPSHOTS[s][0] for s in sns])
    garg = SimpleNamespace(ps_dir=a.ps_dir, err="mode", sys_frac=0.0, kmin=0.05)

    k_all = None
    truth = {"nla": [], "highk": [], "bconst": []}
    terr = []
    meta = {"a1_nla": [], "Ns_nla": [], "Ns_highk": []}
    fits = {"NLA": [], "TATT": []}
    kgrid = None
    for sn in sns:
        z = F.SNAPSHOTS[sn][0]
        D = model.D(z)
        k, PNL, b, sp = combined(a.ps_dir, sn, model)
        if k_all is None:
            k_all = k
        assert np.allclose(k, k_all, rtol=1e-3)
        # NLA side from the nfold-1 data, as the notes
        d1 = F.load_snapshot(a.ps_dir, sn, 1)
        P1 = model.Pnl(z, d1["k"])
        pre = dict(D=D, PNL=P1, bk=F.scale_bias(d1, P1))
        nf = nla_fit(model, d1, pre, None)
        d4 = F.load_snapshot(a.ps_dir, sn, 4)
        sel = (d4["k"] > 5) & (d4["k"] < 10)
        Ns_hk = float(np.median(d4["BB"][0][sel]))
        meta["a1_nla"].append(nf["a1"])
        meta["Ns_nla"].append(nf["Ns"])
        meta["Ns_highk"].append(Ns_hk)
        w = weight(k, KT[sn], SIGMA_W)
        C1 = nf["C1"]
        for tag, Ns in (("nla", nf["Ns"]), ("highk", Ns_hk)):
            RdE = (1 - w)*(-C1) + w*(sp["hE"]/b)/PNL
            REE = (1 - w)*C1**2 + w*(sp["EE"] - Ns)/PNL
            RBB = w*(sp["BB"] - Ns)/PNL
            truth[tag].append(np.array([RdE, REE, RBB]))
        # constant large-scale bias for P_dE = P_hE / b (b averaged at 0.05-0.2)
        b0 = float(np.mean(b[(k >= 0.05) & (k <= 0.2)]))
        RdE = (1 - w)*(-C1) + w*(sp["hE"]/b0)/PNL
        truth["bconst"].append(np.array([RdE, truth["highk"][-1][1], truth["highk"][-1][2]]))
        # 1-sigma FLAMINGO errors of the stitched truth (sim side only)
        terr.append(np.array([w*sp["shE"]/b/PNL, w*sp["sEE"]/PNL, w*sp["sBB"]/PNL]))

        # per-snapshot fits vs k_max (gof_per_snapshot, FLAMINGO error bars)
        s = G.load(model, sn, garg)
        kd = s["k"]
        kms = kd[(kd >= 0.12) & (kd <= 1.75)]
        if kgrid is None:
            kgrid = kms
        Kk = model.kernels(z, k)
        mw = (kd >= 0.05) & (kd <= a.kwide*(1 + 1e-9))
        for mdl in ("NLA", "TATT"):
            rows, warm = [], None
            for km in kms:
                f = G.fit(model, s, mdl, ["hE", "EE"], km, garg, warm)
                warm = f["x"]
                th = dict(a1=0.0, a2=0.0, bTA=0.0, Ns=0.0)
                th.update(f["th"])
                pr = model.predict(z, k, th["a1"], th["a2"], th["bTA"], 1.0, K=Kk,
                                   PNL=PNL, Ns=0.0, bk=b)
                R = np.array([pr["dE"]/PNL, pr["EE"]/PNL, pr["BB"]/PNL])
                # chi2 against the data over the wide range (same errors as the fit)
                pd = model.predict(z, kd, th["a1"], th["a2"], th["bTA"], 1.0, K=s["K"],
                                   PNL=s["PNL"], Ns=th["Ns"], bk=s["bk"])
                cw = sum(float(np.sum((((s["y"][p] - pd[p])/s["err"][p])[mw])**2))
                         for p in ("hE", "EE"))
                rows.append(dict(R=R, chi2=f["chi2"], dof=f["dof"], chi2w=cw,
                                 a1=th["a1"], a2=th["a2"], bTA=th["bTA"], Ns=th["Ns"],
                                 e=f["err"]))
            fits[mdl].append(rows)
        print(f"sn{sn} z={z:.2f}: a1_NLA(lowk)={nf['a1']:.3f} Ns={nf['Ns']:.4f} "
              f"Ns_highk={Ns_hk:.4f}", flush=True)

    out = dict(zs=zs, sns=np.array(sns), k=k_all, kmax=kgrid,
               truth_nla=np.array(truth["nla"]), truth_highk=np.array(truth["highk"]),
               truth_bconst=np.array(truth["bconst"]), truth_err=np.array(terr),
               **{m: np.array(v) for m, v in meta.items()})
    for mdl in fits:
        out[f"R_{mdl}"] = np.array([[r["R"] for r in rows] for rows in fits[mdl]])
        for key in ("chi2", "dof", "chi2w", "a1", "a2", "bTA", "Ns"):
            out[f"{key}_{mdl}"] = np.array([[r[key] for r in rows] for rows in fits[mdl]])
        out[f"a1err_{mdl}"] = np.array([[r["e"].get("a1", 0) for r in rows]
                                        for rows in fits[mdl]])
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
