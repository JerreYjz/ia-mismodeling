#!/usr/bin/env python3
"""Production truth with a physical low-k extension: P_EE below k = 0.05 is
the GI-consistent part (ratio held) plus the white excess X_s kept white,
instead of the whole ratio held. Fits at 0.25, 0.35, 0.5/D."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import bias_analysis as B
import bias_v2 as V
from cosmolike_xi_ia import KW, KMIN, write_set, extend_ratio

def main():
    data = sys.argv[1]; t0 = time.time()
    R = np.load("run/cosmolike_xi_ia.npz"); cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    f = np.load("../v2/fits.npz"); zs, sns, kd = f["zs"], f["sns"], f["k"][0]
    y, err, b, Pf, Ns, T = f["y"], f["err"], f["b"], f["PNL"], f["Ns_true"], f["T"]
    cl = B.CL(os.path.join(HERE, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h; xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    PNLw = np.array([cl.PKNL.P(z, KW*h)*h**3 for z in zs])
    Pf_w = np.exp(np.array([np.interp(np.log(KW), np.log(kd), np.log(Pf[s])) for s in range(13)]))
    sel = (kd >= KMIN) & (kd <= 0.5)
    X_meas = (y[:, 1] - Ns[:, None]) - (y[:, 0]/b)**2/Pf
    dX = np.sqrt(err[:, 1]**2 + (2*(y[:, 0]/b)*err[:, 0]/b/Pf)**2)
    Xw = np.array([np.sum(X_meas[s, sel]/dX[s, sel]**2)/np.sum(1/dX[s, sel]**2) for s in range(13)])
    rdE = T[:, 0]/(b*Pf); rEE = (T[:, 1] - Ns[:, None])/Pf
    PdE, PEE = [], []
    for s in range(13):
        rd = extend_ratio(kd, rdE[s])
        re = extend_ratio(kd, rEE[s] - Xw[s]/Pf[s])          # GI-consistent-like part, ratio held
        re = re + Xw[s]/Pf_w[s]                              # white part stays white everywhere
        PdE.append(rd*PNLw[s]); PEE.append(re*PNLw[s])
    p = write_set("run/tables_diag", "prod_phys", sns, np.array(PdE), np.array(PEE))
    cl.at(B.FID); cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
    xi = cl.xi_sim() - xi_gg
    np.save("run/xi_flam_phys.npy", xi)
    q = lambda v, m: float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))
    print("prod_phys - production: Delta chi2 on masks " + ", ".join(f"{K0}/D {q(xi - R['xi_flam'], M1[K1.index(K0)]):.3f}" for K0 in (0.25, 0.35, 0.5)), flush=True)
    out = []
    for K0 in (0.25, 0.35, 0.5):
        m = M1[K1.index(K0)]; C = cov[np.ix_(m, m)]; Ci = np.linalg.inv(C); Lw = np.linalg.inv(np.linalg.cholesky(C))
        print(f"[prod_phys, mask {K0}/D]", flush=True)
        rn = V.fit(cl, "NLA", xi_gg + xi, Ci, Lw, m, ["omegam", "As_1e9"], print)
        rt = V.fit(cl, "TATT", xi_gg + xi, Ci, Lw, m, ["omegam", "As_1e9"], print, nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]))
        for r in (rn, rt):
            r.update(variant="prod_phys", mask=K0); out.append(r)
        json.dump(out, open("analysis/bias_phys.json", "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")
if __name__ == "__main__":
    main()
