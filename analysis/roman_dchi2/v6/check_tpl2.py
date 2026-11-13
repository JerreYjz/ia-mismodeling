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
Th = HR.templates(cl, tab, False, "halo"); Tn = HR.templates(cl, tab, False, "nla")
sns = np.load("../v2/fits.npz")["sns"]; zs = tab["zs"]; x = (1 + zs)/1.62
S = np.load(os.path.join(HERE, "..", "v3", "masks_scan.npz")); m = S["D1_0.5"].astype(bool); C = cov[np.ix_(m, m)]
def q(v): v = v[m]; return float(v @ np.linalg.solve(C, v))
def run(PdE, PEE):
    p = write_set("run/tables_halo", "check3", sns, PdE, PEE); cl.at(B.FID); cl.use_sim_tables(p, sns, [float(z) for z in zs], 1); return cl.xi_sim() - xi_gg
for label, use_frc, use_f2 in (("NLA A=5.3", False, False), ("f_rc only", True, False), ("f2h only", False, True), ("f_rc and f2h", True, True)):
    A1, eta = 5.3, 0.6
    GI = np.zeros((len(zs), len(KW))); II = np.zeros_like(GI); tg = 0; ti = 0
    for s, z in enumerate(zs):
        Dz = np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05)); C1 = B.FID["omegam"]*H.C1RHOCRIT/Dz
        P = cl.PKNL.P(z, KW*h)*h**3; A = A1*x[s]**eta
        frc = tab["hod"][s, 5] if use_frc else 1.0; w2 = H.f2h(KW) if use_f2 else 1.0
        GI[s] = -A*C1*frc*P*w2; II[s] = (A*C1*frc)**2*P*w2
    fg = run(GI, 0*II); fi = run(0*GI, II)
    if label == "NLA A=5.3":
        As = A1*x**eta; tg = As @ Tn["GI2h"]; ti = (As**2) @ Tn["II2h"]
        print(label, "GI: dchi2", q(fg - tg), " II: dchi2", q(fi - ti), " chi2(GI)", q(fg), "chi2(II)", q(fi), flush=True)
    elif label == "f_rc and f2h":
        As = A1*x**eta; tg = As @ Th["GI2h"]; ti = (As**2) @ Th["II2h"]
        print(label, "GI: dchi2", q(fg - tg), " II: dchi2", q(fi - ti), " chi2(GI)", q(fg), "chi2(II)", q(fi), flush=True)
    else:
        print(label, "chi2(GI)", q(fg), "chi2(II)", q(fi), flush=True)
