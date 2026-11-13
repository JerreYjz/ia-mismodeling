import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); COCOA = os.path.join(HERE, "..", "cocoa")
sys.path.insert(0, COCOA); sys.path.insert(0, HERE)
import bias_analysis as B
from cosmolike_xi_ia import KW, write_set
import halo_ia as H, halo_roman as HR
os.chdir(COCOA)
cov = np.load("analysis/cov_example1_xi.npy"); tab = dict(np.load(os.path.join(HERE, "halo_tables.npz")))
cl = B.CL(os.path.join(COCOA, "lib"), sys.argv[1]); h = cl.h
xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
T = HR.templates(cl, tab, False, "halo"); M = HR.IAModel(T, tab, "halo")
sns = np.load("../v2/fits.npz")["sns"]; zs = tab["zs"]; x = (1 + zs)/1.62
idx = np.searchsorted(tab["k_all"], KW); la = np.log(np.maximum(tab["a_grid"], 1e-6))
S = np.load(os.path.join(HERE, "..", "v3", "masks_scan.npz"))
for ia in ([5.3, 0.6, 0.0, 0.0], [5.3, 0.6, 0.0316, 0.0], [5.3, 0.6, 0.05, 0.0]):
    PdE = np.zeros((len(zs), len(KW))); PEE = np.zeros_like(PdE)
    for s, z in enumerate(zs):
        Dz = np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05)); C1 = B.FID["omegam"]*H.C1RHOCRIT/Dz
        P = cl.PKNL.P(z, KW*h)*h**3; A = ia[0]*x[s]**ia[1]; a = ia[2]*x[s]**ia[3]; frc = tab["hod"][s, 5]
        if a > 0:
            j = int(np.clip(np.searchsorted(la, np.log(a)), 1, la.size - 1)); w = np.clip((np.log(a) - la[j - 1])/(la[j] - la[j - 1]), 0, 1)
            Sd = tab["SdI"][s, j - 1][idx]*(1 - w) + tab["SdI"][s, j][idx]*w; Si = tab["SII"][s, j - 1][idx]*(1 - w) + tab["SII"][s, j][idx]*w
        else:
            Sd = Si = 0*KW
        PdE[s] = -(A*C1*frc*P*H.f2h(KW) + a*H.f1h(KW)*Sd); PEE[s] = (A*C1*frc)**2*P*H.f2h(KW) + a**2*H.f1h(KW)*Si
    p = write_set("run/tables_halo", "check2", sns, PdE, PEE); cl.at(B.FID); cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
    full = cl.xi_sim() - xi_gg; tsum = M.xi(ia)
    msg = []
    for key in ("D1_0.3", "D1_0.5", "flat_0.5"):
        m = S[key].astype(bool); d = (full - tsum)[m]; one = (tsum - M.xi([ia[0], ia[1], 0, 0]))[m]
        C = cov[np.ix_(m, m)]
        msg.append(f"{key}: dchi2(full-tpl) {float(d @ np.linalg.solve(C, d)):.2e}, chi2(1-halo) {float(one @ np.linalg.solve(C, one)):.2e}")
    print(ia, "; ".join(msg), flush=True)
