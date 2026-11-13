#!/usr/bin/env python3
"""Figure of bias: Roman cosmic shear (xi_+-) fitted with NLA / TATT when the
data carry FLAMINGO's IA.

Data:   d = xi_GG(D3A) + xi_IA(FLAMINGO), FLAMINGO's P_dE and P_EE injected
        through sim_IA (conversion verified, verify_injection.py).
Model:  cosmolike's own NLA (A1, eta1) or TATT (A1, eta1, A2, eta2, b_TA),
        IA_redshift_evolution = 3 (power law in (1+z)/1.62), as in the chains;
        cosmology Omega_m, A_s (S8 derived), n_s, H0, Omega_b.
Covariance: Gaussian xi_+- (analysis/cov_gauss_d3a.npy), on the production
        IR/UV masks k_UV/D(z).
Fit:    IA exact (cosmolike), cosmology through its linear response
        (central differences with CAMB), refreshed at the IA solution, trust-
        region least squares; exact chi2 re-evaluated at the solution; IA
        parameters flat (a sigma-50 term regularises only the Fisher inverse). Two cosmology settings:
          A  Omega_m, A_s free; n_s, H0, Omega_b fixed (stringent limit)
          B  all five free; n_s, H0, Omega_b with Gaussian priors of the
             width of the yaml flat priors (range / sqrt 12)
Output: (Omega_m, S8) shift at the best fit in units of the marginalised
        errors, figure of bias FoB = sqrt(dtheta^T Sigma_theta^-1 dtheta)
        (criterion: < 0.3), chi2 at the best fit (Delta chi2_Roman after the
        model absorbs what it can), and two null tests: TATT fitted to a TATT
        data vector (Part 1a fiducial) must give zero bias, and NLA fitted to
        it gives the Part 1 NLA-on-TATT bias.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cosmolike_driver import Cosmolike, D3A  # noqa: E402

CP = ["omegam", "As_1e9", "ns", "H0", "omegab"]
FID = dict(omegam=0.306, As_1e9=2.099, ns=0.967, H0=68.1, omegab=0.0486)
STEP = dict(omegam=0.01, As_1e9=0.05, ns=0.01, H0=1.0, omegab=0.002)
PRIOR_B = dict(ns=0.2/np.sqrt(12), H0=36/np.sqrt(12), omegab=0.04/np.sqrt(12))
IAP = {"NLA": ["A1", "eta1"], "TATT": ["A1", "eta1", "A2", "eta2", "bTA"]}
IA_START = {"NLA": [3.4, 0.9], "TATT": [3.8, 0.57, -1.08, -0.71, 0.18]}
TATT_FID = [3.80, 0.57, -1.08, -0.71, 0.18]      # Part 1a fiducial
MASKS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}


class CL(Cosmolike):
    """Cosmolike driver with cached CAMB states per cosmology."""
    cache = {}
    order = []

    def at(self, p):
        c = dict(D3A)
        c.update(omegam=p["omegam"], As=p["As_1e9"]*1e-9, ns=p["ns"], H0=p["H0"], omegab=p["omegab"])
        key = (round(c["omegam"], 8), round(c["As"]*1e9, 8), round(c["ns"], 8), round(c["H0"], 6),
               round(c["omegab"], 8))
        self.cosmo = c
        if key not in self.cache:
            self._camb()
            self.cache[key] = (self.res, self.h, self.omegan2, self.PKL, self.PKNL, self.PKLcb, self.sigma8)
            self.order.append(key)
            if len(self.order) > 40:                 # bounded cache (memory)
                self.cache.pop(self.order.pop(0), None)
        (self.res, self.h, self.omegan2, self.PKL, self.PKNL, self.PKLcb, self.sigma8) = self.cache[key]
        return self

    def model(self, mdl, p, ia):
        self.at(p)
        if mdl == "NLA":
            return self.xi_native(0, A1=[ia[0], ia[1]], A2=[0.0, 0.0], BTA=[0.0])
        return self.xi_native(1, A1=[ia[0], ia[1]], A2=[ia[2], ia[3]], BTA=[ia[4]])

    def s8(self, p):
        self.at(p)
        return self.sigma8*np.sqrt(p["omegam"]/0.3)


def fit(cl, mdl, d, Ci, Lw, m, free, prior, log):
    """Best fit of (free cosmology + IA) to d on mask m.

    IA enters exactly (cosmolike at the fiducial cosmology); cosmology through
    its linear response J_c(ia) = d m / d p_c (central differences, CAMB at
    p_fid +- step, cached), refreshed at the IA solution and the fit redone:
        m(p_c, ia) ~= m(fid, ia) + J_c(ia) (p_c - fid)
    solved with scipy least_squares (trust region). The exact model is then
    evaluated at the solution (CAMB at the best-fit cosmology) and its chi2
    reported next to the linearised one."""
    nia = len(IAP[mdl])
    names = free + IAP[mdl]
    pri = np.array([prior.get(n, np.inf) for n in free] + [50.0]*nia)    # 50: Fisher regulariser only
    x_fid = np.concatenate([[FID[n] for n in free], np.zeros(nia)])

    def ia_model(ia):
        return cl.model(mdl, FID, ia)[m]

    def Jc_at(ia):
        J = np.zeros((m.sum(), len(free)))
        for i, n in enumerate(free):
            pp, pm = dict(FID), dict(FID)
            pp[n] += STEP[n]
            pm[n] -= STEP[n]
            J[:, i] = (cl.model(mdl, pp, ia) - cl.model(mdl, pm, ia))[m]/(2*STEP[n])
        return J

    # IA-only fit at the fiducial cosmology
    r0 = least_squares(lambda ia: Lw @ (d[m] - ia_model(ia)), np.array(IA_START[mdl]), x_scale="jac",
                       max_nfev=600)
    ia0 = r0.x
    chi2_ia = float(r0.fun @ r0.fun)
    x = np.concatenate([[FID[n] for n in free], ia0])
    for rep in range(2):
        Jc = Jc_at(x[len(free):])
        def resid(x):
            dc = x[:len(free)] - x_fid[:len(free)]
            mod = ia_model(x[len(free):]) + Jc @ dc
            # Gaussian cosmology priors (case B) centred on the fiducial; IA flat
            rp = np.where(np.isfinite(pri[:len(free)]), dc/np.where(np.isfinite(pri[:len(free)]),
                                                                    pri[:len(free)], 1.0), 0.0)
            return np.concatenate([Lw @ (d[m] - mod), rp])
        r = least_squares(resid, x, x_scale="jac", max_nfev=2000)
        x = r.x
    chi2_lin = float(np.sum((Lw @ (d[m] - ia_model(x[len(free):]) - Jc @ (x[:len(free)] - x_fid[:len(free)])))**2))
    pc = dict(FID)
    pc.update(zip(free, x[:len(free)]))
    ia = x[len(free):]
    res_exact = (d - cl.model(mdl, pc, ia))[m]
    chi2 = float(res_exact @ Ci @ res_exact)
    # Fisher at the solution: cosmology from J_c, IA by central differences
    Jia = np.zeros((m.sum(), nia))
    for i in range(nia):
        e = np.zeros(nia)
        e[i] = 0.05
        Jia[:, i] = (ia_model(ia + e) - ia_model(ia - e))/0.1
    J = np.hstack([Jc, Jia])
    Pinv = np.diag([0.0 if not np.isfinite(v) else 1/v**2 for v in pri])
    Cov = np.linalg.inv(J.T @ Ci @ J + Pinv)
    s8_fid, s8_fit = cl.s8(FID), cl.s8(pc)
    G = np.zeros((2, len(names)))
    G[0, names.index("omegam")] = 1.0
    for i, n in enumerate(free):
        pp, pm = dict(FID), dict(FID)
        pp[n] += STEP[n]
        pm[n] -= STEP[n]
        G[1, i] = (cl.s8(pp) - cl.s8(pm))/(2*STEP[n])
    St = G @ Cov @ G.T
    dth = np.array([pc["omegam"] - FID["omegam"], s8_fit - s8_fid])
    fob = float(np.sqrt(dth @ np.linalg.solve(St, dth)))
    out = dict(model=mdl, free=free, ia0=ia0.tolist(), chi2_ia_only=chi2_ia, chi2_lin=chi2_lin,
               chi2_bestfit=chi2, best=dict(zip(names, x.tolist())), dOm=float(dth[0]), dS8=float(dth[1]),
               sOm=float(np.sqrt(St[0, 0])), sS8=float(np.sqrt(St[1, 1])),
               rho=float(St[0, 1]/np.sqrt(St[0, 0]*St[1, 1])), FoB=fob, npts=int(m.sum()),
               S8_fid=float(s8_fid))
    log(f"    {mdl:4s} [{','.join(free)}]: chi2 IA-only {chi2_ia:6.2f}, best fit {chi2:6.2f} "
        f"(linearised {chi2_lin:6.2f}); dOm {dth[0]:+.4f} ({dth[0]/out['sOm']:+.2f} s), dS8 {dth[1]:+.4f} "
        f"({dth[1]/out['sS8']:+.2f} s), rho {out['rho']:+.2f}, FoB {fob:.2f}; IA {np.round(ia, 3)}")
    return out


def main():
    data = sys.argv[1]
    t0 = time.time()
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    cl = CL(os.path.join(HERE, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (
        cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    xi_gg = cl.model("NLA", FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    d_tatt = cl.model("TATT", FID, TATT_FID)
    logf = open("analysis/bias_log.txt", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    log(f"setup {time.time() - t0:.0f}s; S8(D3A) = {cl.s8(FID):.4f}")
    results = []
    for K0 in MASKS:
        m = M1[K1.index(K0)]
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        Lw = np.linalg.inv(np.linalg.cholesky(C))
        log(f"mask {K0}/D(z): {int(m.sum())} points  ({time.time() - t0:.0f}s)")
        for case, free, prior in (("A", ["omegam", "As_1e9"], {}),
                                  ("B", CP, PRIOR_B)):
            for mdl in ("NLA", "TATT"):
                r = fit(cl, mdl, d_fl, Ci, Lw, m, free, prior, log)
                r.update(data="FLAMINGO", case=case, mask=K0)
                results.append(r)
            if K0 in (0.25, 0.35, 0.5):
                for mdl in ("TATT", "NLA"):
                    log("    [null/Part 1 check: data = TATT fiducial]")
                    r = fit(cl, mdl, d_tatt, Ci, Lw, m, free, prior, log)
                    r.update(data="TATT_fid", case=case, mask=K0)
                    results.append(r)
        json.dump(results, open("analysis/bias_results.json", "w"), indent=1)
    logf.close()
    plots(results)
    print(f"done ({time.time() - t0:.0f}s)")


def plots(results):
    for case in ("A", "B"):
        fig, axs = plt.subplots(1, 3, figsize=(21, 6.3))
        for mdl in COL:
            rr = [r for r in results if r["data"] == "FLAMINGO" and r["case"] == case and r["model"] == mdl]
            K = [r["mask"] for r in rr]
            axs[0].plot(K, [r["FoB"] for r in rr], "o-", color=COL[mdl], lw=2, label=mdl)
            axs[1].plot(K, [r["dOm"]/r["sOm"] for r in rr], "o-", color=COL[mdl], label=f"{mdl}: Omega_m")
            axs[1].plot(K, [r["dS8"]/r["sS8"] for r in rr], "s--", color=COL[mdl], label=f"{mdl}: S8")
            axs[2].plot(K, [r["chi2_bestfit"] for r in rr], "o-", color=COL[mdl], label=f"{mdl}: best fit (cosmology + IA free)")
            axs[2].plot(K, [r["chi2_ia_only"] for r in rr], "s:", color=COL[mdl], label=f"{mdl}: IA only, cosmology fixed")
            tt = [r for r in results if r["data"] == "TATT_fid" and r["case"] == case and r["model"] == mdl]
            if tt:
                axs[0].plot([r["mask"] for r in tt], [r["FoB"] for r in tt], "x:", color=COL[mdl], ms=9,
                            label=f"{mdl} on TATT fiducial (null for TATT)")
        axs[0].axhline(0.3, color="k", lw=1)
        axs[0].set_ylabel("figure of bias in (Omega_m, S8)  [sigma, 2D]")
        axs[0].set_title("FoB; line: 0.3 sigma criterion")
        axs[1].axhline(0, color="k", lw=0.7)
        for v in (-0.3, 0.3):
            axs[1].axhline(v, color="k", lw=0.7, ls=":")
        axs[1].set_ylabel("shift / marginalised sigma")
        axs[1].set_title("1D shifts at the best fit")
        axs[2].axhline(1, color="k", lw=1)
        axs[2].set_yscale("log")
        axs[2].set_ylabel(r"$\chi^2$ at the best fit (Roman, masked $\xi_\pm$)")
        axs[2].set_title("what the model cannot absorb")
        for ax in axs:
            ax.set_xlabel(r"IR/UV mask $k_{UV}/D(z)$, $k_{UV}\ [h\,{\rm Mpc}^{-1}]$")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        lab = {"A": "Omega_m, A_s free (n_s, H0, Omega_b fixed)",
               "B": "Omega_m, A_s, n_s, H0, Omega_b free (flat-prior widths as Gaussians)"}[case]
        fig.suptitle(f"Roman cosmic shear, data = GG + FLAMINGO IA; model = cosmolike NLA / TATT (power-law z "
                     f"evolution); {lab}", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"analysis/fig_bias_{case}.png", dpi=220)
        fig.savefig(f"analysis/fig_bias_{case}.pdf")
        plt.close(fig)

        # ellipses in (Omega_m, S8) for three masks
        fig, axs = plt.subplots(1, 3, figsize=(19, 6.3))
        for ax, K0 in zip(axs, (0.25, 0.35, 0.5)):
            for mdl in COL:
                r = [x for x in results if x["data"] == "FLAMINGO" and x["case"] == case and x["model"] == mdl
                     and x["mask"] == K0][0]
                so, ss, rho = r["sOm"], r["sS8"], r["rho"]
                Cm = np.array([[so**2, rho*so*ss], [rho*so*ss, ss**2]])
                ev, V = np.linalg.eigh(Cm)                     # ascending; V[:, 1] = major axis
                for nsig, a in ((1, 0.25), (2, 0.12)):
                    c2 = 2.30 if nsig == 1 else 6.18             # 2D 68% / 95%
                    w, hgt = 2*np.sqrt(c2*ev[1]), 2*np.sqrt(c2*ev[0])
                    ell = Ellipse((r["dOm"], r["dS8"]), w, hgt, angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])),
                                  color=COL[mdl], alpha=a, label=f"{mdl} 68/95%" if nsig == 1 else None)
                    ax.add_patch(ell)
                ax.plot(r["dOm"], r["dS8"], "o", color=COL[mdl], mec="k",
                        label=f"{mdl}: FoB {r['FoB']:.2f}, chi2 {r['chi2_bestfit']:.1f}")
            ax.plot(0, 0, "k*", ms=14, label="FLAMINGO D3A truth")
            ax.set_xlabel(r"$\Delta\Omega_m$")
            ax.set_ylabel(r"$\Delta S_8$")
            ax.set_title(f"mask {K0}/D(z)")
            ax.autoscale_view()
            lim = max(4*max(r["sOm"] for r in results if r["mask"] == K0 and r["case"] == case), 0.01)
            ax.set_xlim(-lim, lim)
            lim8 = max(4*max(r["sS8"] for r in results if r["mask"] == K0 and r["case"] == case), 0.005)
            ax.set_ylim(-lim8, lim8)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.legend(fontsize=8)
        fig.suptitle(f"Best-fit shift and marginalised contours, case {case}", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"analysis/fig_bias_ellipses_{case}.png", dpi=220)
        fig.savefig(f"analysis/fig_bias_ellipses_{case}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--plots-only":
        plots(json.load(open("analysis/bias_results.json")))
    else:
        main()
