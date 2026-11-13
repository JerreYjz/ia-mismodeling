#!/usr/bin/env python3
"""NLA / TATT plus one free amplitude a_w for a white II template (the Roman
xi of the per-snapshot white excess X_s, cosmolike), fitted to the production
FLAMINGO truth. If the FoB drops to the single-tidal-field values, the common
NLA/TATT bias is the white II term and nothing else."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bias_analysis as B
import bias_v2 as V

TPL = np.load("run/xi_white_only.npz")["ratio_ratio"]
B.IAP["NLAw"] = ["A1", "eta1", "aw"]
B.IAP["TATTw"] = ["A1", "eta1", "A2", "eta2", "bTA", "aw"]
V.BOUNDS["aw"] = (-5, 5)
V.IA_SIG["aw"] = 10/np.sqrt(12)
_orig = B.CL.model
def model(self, mdl, p, ia):
    if mdl == "NLAw":
        return _orig(self, "NLA", p, ia[:2]) + ia[2]*TPL
    if mdl == "TATTw":
        return _orig(self, "TATT", p, ia[:5]) + ia[5]*TPL
    return _orig(self, mdl, p, ia)
B.CL.model = model
_starts = V.starts
def starts(mdl, nla_sol):
    if mdl == "NLAw":
        return [np.array([nla_sol[0], nla_sol[1], 1.0]), np.array([3.4, 0.9, 0.0])]
    if mdl == "TATTw":
        return [np.concatenate([s, [1.0]]) for s in _starts("TATT", nla_sol)]
    return _starts(mdl, nla_sol)
V.starts = starts

def main():
    data = sys.argv[1]; t0 = time.time()
    R = np.load("run/cosmolike_xi_ia.npz"); cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    cl = B.CL(os.path.join(HERE, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0]); d = xi_gg + R["xi_flam"]
    out = []
    for K0 in (0.25, 0.35, 0.5):
        m = M1[K1.index(K0)]; C = cov[np.ix_(m, m)]; Ci = np.linalg.inv(C); Lw = np.linalg.inv(np.linalg.cholesky(C))
        print(f"[white template, mask {K0}/D]", flush=True)
        rn = V.fit(cl, "NLAw", d, Ci, Lw, m, ["omegam", "As_1e9"], print)
        rt = V.fit(cl, "TATTw", d, Ci, Lw, m, ["omegam", "As_1e9"], print, nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]))
        for r in (rn, rt):
            r.update(mask=K0); out.append(r)
        json.dump(out, open("analysis/bias_wtemplate.json", "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")
if __name__ == "__main__":
    main()
