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
Th = HR.templates(cl, tab, False, "halo")
sns = np.load("../v2/fits.npz")["sns"]; zs = tab["zs"]; x = (1 + zs)/1.62
S = np.load(os.path.join(HERE, "..", "v3", "masks_scan.npz")); m = S["D1_0.5"].astype(bool); C = cov[np.ix_(m, m)]
def q(v): v = v[m]; return float(v @ np.linalg.solve(C, v))
def run(PdE, PEE):
    p = write_set("run/tables_halo", "check4", sns, PdE, PEE); cl.at(B.FID); cl.use_sim_tables(p, sns, [float(z) for z in zs], 1); return cl.xi_sim() - xi_gg
zero = np.zeros((len(zs), len(KW)))
fresh = []
for s, z in enumerate(zs):
    Dz = np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05)); C1 = B.FID["omegam"]*H.C1RHOCRIT/Dz
    P = cl.PKNL.P(z, KW*h)*h**3; frc = tab["hod"][s, 5]
    b = zero.copy(); b[s] = (C1*frc)**2*P*H.f2h(KW)
    fresh.append(run(zero, b))
fresh = np.array(fresh)
for s in range(len(zs)):
    print(s, "dchi2(fresh - cached template)", q(fresh[s] - Th["II2h"][s]), " chi2(template)", q(Th["II2h"][s]), flush=True)
