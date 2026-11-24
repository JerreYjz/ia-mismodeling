#!/usr/bin/env python3
"""Stage 1 of the v9 IA error budget for 3x2pt: the base with gamma_t and w.

The same knobs as ia_base.py (a fractional error of P_X, X = GI, II, on one
ln k tent at one snapshot, 2 x 13 x 61 = 1586 per truth), with the response
written on the full Roman 3x2pt vector of cocoa/driver3x2.py (2115 points:
xi_+ | xi_- | gamma_t, 61 lens-source pairs | w, 8 lens bins). IA enters
gamma_t only through the GI term, b_i P_dE at the lens redshifts, with the
same FLAMINGO P_dE (sim_IA tables) as xi_+-; II enters xi_+- alone and w has
no IA. So the GI knobs are shared by the two probes, which is the physics:
one P_dE error moves both. The lenses share the source n(z), with linear
bias b_i = B1_FID, Limber everywhere (as v3/ggl3x2.py).

Stored per truth (ia_base_3x2_<truth>.npz, the layout of ia_base.py):
  J      d dv/d(Omega_m, A_s, A, eta, b_1..b_8) of the truth-as-model,
         central differences (the biases flat in the fits);
  g_s8   the (Omega_m, S8) rows, zero on the biases;
  R_GI, R_II, xi0 (the full data vector), xi_ia (its IA part);
  checks the tents summed against xi_ia; the xi_+- rows of every column and
         of J against ia_base_<truth>.npz (must agree to round-off); II
         columns zero on gamma_t and w.
ia_base_3x2_common.npz: the Gaussian 3x2pt covariance (cocoa/cov3x2.py; the
shipped example1 3x2pt matrix is not positive definite: its xi-gamma_t
block is inconsistent), the lens-source pairs, the masks per probe
(xi|<K0>: masks_n1 of run/cosmolike_xi_ia.npz; gt|<K0> and w|<K0>: the
alpha = 0.05 IR/UV rows of k_UV = K0/D(z) on the 3x2pt vector, as
v3/ggl3x2.py), "behind" (gamma_t rows with the source bin behind the lens
bin), and the composed cuts used by ia_budget_3x2.py:
  cs|<K>    xi_+- alone with masks_n1 at K (the cosmic-shear analysis);
  3x2|<K>   xi_+- and gamma_t at K0 = K, w at K0 = 0.2 (every pair);
  3x2b|<K>  the same, gamma_t of sources behind lenses only.

    python ia_base_3x2.py <cosmolike so dir> <cocoa_roman_real/data> [--truths S1,prod]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(HERE, "..", "v8")
sys.path.insert(0, V8)
sys.path.insert(0, HERE)
import clean_budget as CB                                          # noqa: E402
from clean_budget import T, B, KW                                  # noqa: E402
from tolerance_bands import write_set                              # noqa: E402
import ia_fisher as IF                                             # noqa: E402
import bmode_tables as BT                                          # noqa: E402
sys.path.insert(0, T.TB.COCOA)
from driver3x2 import Cosmolike3x2, B1_FID                         # noqa: E402
from cosmolike_xi_ia import LN_FLOOR, mask_rows                    # noqa: E402

SCRATCH = "v9_scratch_3x2"
NXI, NGT, NDV = 1080, 915, 2115
BN = [f"b{i + 1}" for i in range(8)]
FREE_T, IAP = ["omegam", "As_1e9"], ["A", "eta"]
K0_GGL = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8)
K0_GC = 0.2
BIAS_STEP = 0.02


class CL3(Cosmolike3x2):
    """The 3x2pt driver with the cached-CAMB cosmology of bias_analysis.CL."""
    cache = B.CL.cache
    order = B.CL.order
    at = B.CL.at
    s8 = B.CL.s8


class Truth3(CB.Truth):
    """The truth-as-model on the 3x2pt vector: cosmology, (A, eta), b_i."""

    def __call__(self, p, ia):
        path = self.tables(p, ia, "template3")
        cl = self.cl.at(p)
        cl.use_sim_tables(path, self.sns, self.zl, 1)
        self.n += 1
        return cl.dv_sim(B1=[p[n] for n in BN])


def dv_ia(cl, model, pde, pee, gg):
    """The IA part of the 3x2pt vector for given tables, at the fiducial."""
    path = write_set(T.TB.TABLES, SCRATCH, model.sns, pde, pee)
    cl.at(B.FID)
    cl.use_sim_tables(path, model.sns, model.zl, 1)
    return cl.dv_sim() - gg


def ir_masks(cl, K0s):
    """{K0: bool (2115,)}: the alpha = 0.05 IR/UV rows of k_UV = K0/D(z) on
    the 3x2pt vector (v3/ggl3x2.py, GG + NLA baseline), every probe."""
    cl.at(B.FID)
    cl.ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
    cl.sim = False
    h = cl.h
    k = 10.0**cl.lk/h
    z2 = cl.z2
    lnPL = np.log(cl.PKL.P(z2, 10.0**cl.lk)) + np.log(h**3)
    lnPNL = np.log(cl.PKNL.P(z2, 10.0**cl.lk)) + np.log(h**3)
    lnPLcb = np.log(cl.PKLcb.P(z2, 10.0**cl.lk)) + np.log(h**3)
    zg = cl.z1[cl.z1 <= z2[-1]]
    G = np.sqrt(cl.PKL.P(zg, 0.05)/cl.PKL.P(0, 0.05))*(1 + zg)
    G = np.asarray(G/(np.sqrt(cl.PKL.P(z2[-1], 0.05)/cl.PKL.P(0, 0.05))*(1 + z2[-1]))).ravel()
    Dz2 = np.sqrt(np.asarray(cl.PKL.P(z2, 1e-3))/cl.PKL.P(0.0, 1e-3)).ravel()

    def dv_with(lnL, lnN):
        cl.ci.set_cosmology(omegam=cl.cosmo["omegam"], omegab=cl.cosmo["omegab"], omegan2=cl.omegan2,
                            H0=cl.cosmo["H0"], log10k_2D=cl.lk - np.log10(h), z_2D=z2,
                            lnP_linear=lnL.flatten(order="F"), lnP_linear_cb=lnPLcb.flatten(order="F"),
                            lnP_nonlinear=lnN.flatten(order="F"), G=G, z_G=zg, z_1D=cl.z1,
                            chi=cl.res.comoving_radial_distance(cl.z1)*h)
        cl._nuisance([0.6, -1.5], [0.0], [0.0])
        return np.asarray(cl.ci.compute_data_vector_masked(), dtype=float)

    full = dv_with(lnPL, lnPNL)
    out = {}
    for K0 in K0s:
        keep = k[None, :] <= (K0/Dz2)[:, None]
        ir = dv_with(np.where(keep, lnPL, lnPL + LN_FLOOR), np.where(keep, lnPNL, lnPNL + LN_FLOOR))
        uv = full - ir
        den = np.abs(uv) + np.abs(ir)
        f = np.where(den > 0, np.abs(uv)/np.where(den > 0, den, 1), 0)
        out[K0] = mask_rows(f)
    return out


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    truths = CB.TRUTHS
    if "--truths" in sys.argv:
        truths = tuple(sys.argv[sys.argv.index("--truths") + 1].split(","))
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    logf = open(os.path.join(HERE, "ia_base_3x2_log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    for i, n in enumerate(BN):
        B.FID[n], B.STEP[n] = B1_FID[i], BIAS_STEP
    cl = CL3(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    cl.at(B.FID)
    gg = cl.dv_native(0, A1=[0.0, 0.0], A2=[0.0, 0.0], BTA=[0.0])
    pairs = cl.gs_pairs()
    assert len(pairs)*15 == NGT

    # ---------------- masks and covariance ----------------
    R0 = np.load("run/cosmolike_xi_ia.npz")
    K_cs = [round(float(k), 4) for k in R0["K_n1"]]
    xi_masks = {"none": np.ones(NXI, bool)}
    xi_masks.update({f"{k:g}": m for k, m in zip(K_cs, R0["masks_n1"])})
    K_all = sorted(set(K_cs) | set(K0_GGL) | {K0_GC})
    ir = ir_masks(cl, K_all)
    agree = [int(np.sum(ir[k][:NXI] != xi_masks[f"{k:g}"])) for k in K_cs]
    log(f"[{time.time() - t0:.0f}s] masks: xi rows of the 3x2pt IR/UV masks differ from masks_n1 "
        f"in {agree} points (K0 {K_cs})")
    gt_masks = {"none": np.ones(NGT, bool)}
    gt_masks.update({f"{k:g}": ir[k][NXI:NXI + NGT] for k in K_all})
    w_masks = {"none": np.ones(NDV - NXI - NGT, bool)}
    w_masks.update({f"{k:g}": ir[k][NXI + NGT:] for k in K_all})
    behind = np.zeros(NGT, bool)
    for p, (l, s) in enumerate(pairs):
        behind[15*p:15*(p + 1)] = s > l

    def compose(kx, kg, kw, beh=False, probes=("xi", "gt", "w")):
        m = np.zeros(NDV, bool)
        if "xi" in probes:
            m[:NXI] = xi_masks[kx]
        if "gt" in probes:
            m[NXI:NXI + NGT] = gt_masks[kg] & (behind if beh else True)
        if "w" in probes:
            m[NXI + NGT:] = w_masks[kw]
        return m

    cuts = {"cs|none": compose("none", None, None, probes=("xi",)),
            "3x2|none": compose("none", "none", "none"),
            "3x2b|none": compose("none", "none", "none", beh=True)}
    for k in K_cs:
        key = f"{k:g}"
        cuts[f"cs|{key}"] = compose(key, None, None, probes=("xi",))
        cuts[f"3x2|{key}"] = compose(key, key, f"{K0_GC:g}")
        cuts[f"3x2b|{key}"] = compose(key, key, f"{K0_GC:g}", beh=True)
    cov = np.load("analysis/cov3x2_gauss_d3a.npy")
    assert cov.shape == (NDV, NDV) and np.linalg.eigvalsh(cov).min() > 0
    np.savez(os.path.join(HERE, "ia_base_3x2_common.npz"), cov=cov, pairs=np.array(pairs),
             behind=behind, cuts=np.array(list(cuts)), K_cs=np.array(K_cs), K_ggl=np.array(K_all),
             **{f"mask|{c}": m for c, m in cuts.items()},
             **{f"xi|{k}": m for k, m in xi_masks.items()},
             **{f"gt|{k}": m for k, m in gt_masks.items()},
             **{f"w|{k}": m for k, m in w_masks.items()})
    log(f"[{time.time() - t0:.0f}s] cuts: " + ", ".join(f"{c} {int(m.sum())}" for c, m in cuts.items()))

    st = BT.load_truths(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    Hf = CB.tents_on(KW, CB.FINE)
    x_fid = {**{n: B.FID[n] for n in FREE_T}, "A": 1.0, "eta": 0.0, **{n: B.FID[n] for n in BN}}
    for tr in truths:
        model = Truth3(cl, st, tr)
        zs = model.zs
        nz, nf = len(zs), len(CB.FINE)
        P = model.pnl(B.FID)
        PdE, PEE = model.RdE*P, model.REE*P
        zero = np.zeros_like(PdE)
        xi_ia = dv_ia(cl, model, PdE, PEE, gg)
        withb = BT.has_b(st, tr)
        full = model
        if withb:
            # B-mode II: a knob of II moves E and B together (as ia_base.py)
            mB = Truth3(cl, st, tr + "_B")
            PBB = mB.REE*P
            xi_ia = xi_ia + BT.flip_b(dv_ia(cl, mB, zero, PBB, gg))
            RB = np.zeros((nz, nf, NDV))
            full = BT.EBModel(model, mB, Truth3(cl, st, "ZERO"))
        R = {X: np.zeros((nz, nf, NDV)) for X in IF.XS}
        tl = time.time()
        for s in range(nz):
            for j in range(nf):
                for X in IF.XS:
                    tab = zero.copy()
                    tab[s] = (PdE if X == "GI" else PEE)[s]*Hf[j]
                    R[X][s, j] = dv_ia(cl, model, tab if X == "GI" else zero,
                                       tab if X == "II" else zero, gg)
                if withb:
                    tab = zero.copy()
                    tab[s] = PBB[s]*Hf[j]
                    RB[s, j] = BT.flip_b(dv_ia(cl, mB, zero, tab, gg))
        log(f"[{time.time() - t0:.0f}s] {tr}: {(3 if withb else 2)*nz*nf} columns in {time.time() - tl:.0f}s")
        extra = {}
        if withb:
            extra = dict(R_II_E=R["II"], R_II_B=RB)
            R["II"] = R["II"] + RB
        tab = zero.copy()
        tab[nz//2] = PdE[nz//2]*Hf[nf//2]
        again = dv_ia(cl, model, tab, zero, gg)
        stale = float(np.max(np.abs(again - R["GI"][nz//2, nf//2]))/np.max(np.abs(again)))

        # Jacobian of the truth-as-model: cosmology, (A, eta), biases
        names = FREE_T + IAP + BN
        J = np.zeros((NDV, len(names)))
        for i, n in enumerate(names):
            step = T.STEP_IA[n] if n in IAP else B.STEP[n]
            xs = []
            for sg in (1, -1):
                x = dict(x_fid)
                x[n] += sg*step
                p = dict(B.FID)
                p.update({k: x[k] for k in FREE_T + BN})
                xs.append(full(p, {k: x[k] for k in IAP}))
            J[:, i] = (xs[0] - xs[1])/(2*step)
        g = np.hstack([T.s8_row(cl, FREE_T, IAP, {k: x_fid[k] for k in FREE_T + IAP}),
                       np.zeros((2, len(BN)))])

        v9 = np.load(os.path.join(HERE, f"ia_base_{tr}.npz"))
        lnx = np.log((1 + zs)/IF.X0)
        tot = R["GI"].sum((0, 1)) + R["II"].sum((0, 1))
        scale = np.max(np.abs(xi_ia))
        rel = lambda a, b: float(np.max(np.abs(a - b))/np.max(np.abs(b)))
        checks = dict(
            stale=stale,
            sum_vs_xi_ia=rel(tot, xi_ia),
            xi_rows_vs_v9_GI=rel(R["GI"][..., :NXI], v9["R_GI"]),
            xi_rows_vs_v9_II=rel(R["II"][..., :NXI], v9["R_II"]),
            xi_ia_vs_v9=rel(xi_ia[:NXI], v9["xi_ia"]),
            J_xi_vs_v9=max(rel(J[:NXI, i], v9["J"][:, i]) for i in range(4)),
            II_off_xi=float(np.max(np.abs(R["II"][..., NXI:]))/scale),
            w_ia=float(np.max(np.abs(xi_ia[NXI + NGT:]))),
            sum_vs_dA=rel(R["GI"].sum((0, 1)) + 2*R["II"].sum((0, 1)), J[:, 2]),
            sum_vs_deta=rel((lnx[:, None]*(R["GI"].sum(1) + 2*R["II"].sum(1))).sum(0), J[:, 3]))
        log(f"[{time.time() - t0:.0f}s] {tr}: checks " + ", ".join(f"{k} {v:.1e}" for k, v in checks.items()))
        np.savez(os.path.join(HERE, f"ia_base_3x2_{tr}.npz"), truth=tr, zs=zs, fine=CB.FINE,
                 xi0=gg + xi_ia, xi_ia=xi_ia, J=J, names=np.array(names), template=np.array(IF.TEMPLATE),
                 g_s8=g, R_GI=R["GI"], R_II=R["II"], checks=json.dumps(checks), **extra)
        log(f"[{time.time() - t0:.0f}s] {tr}: wrote ia_base_3x2_{tr}.npz")
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
