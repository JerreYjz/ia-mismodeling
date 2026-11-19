#!/usr/bin/env python3
"""Halo-model IA templates for the minimizer (fisher_minimize.py).

The halo model with a free share alpha of nonlinear power in the 2-halo
term (notes/flamingo_vs_tng300, the refit "alpha free, a_c <= 1, no
satellites") puts the tidal-alignment (A1) terms on

    P_GI = I_1 P_lin + alpha (P_NL - I_1 P_lin),
    P_II = P_lin + alpha (P_NL - P_lin),

where NLA and TATT put them on P_NL. Its A1 terms are therefore those of
cosmolike's native NLA or TATT plus

    dP_GI(k, z) = (1 - alpha) A(z) C1(z) [P_NL - I_1 P_lin]       (sign: GI < 0)
    dP_II(k, z) = -(1 - alpha) A(z)^2 C1(z)^2 [P_NL - P_lin],

with A(z) = A1 ((1 + z)/1.62)^eta1. The TATT terms in A2 and b_TA are
unchanged. This script writes, for every snapshot s of the v9 base, the
data-vector change per unit (1 - alpha) A_s (GI) and (1 - alpha) A_s^2 (II):

    U_GI[s] = sum_j R_GI[s, j] C1_s (1 - I_1 P_lin/P_NL)(k_j, z_s) / R_dE(k_j, z_s)
    U_II[s] = -sum_j R_II[s, j] C1_s^2 (1 - P_lin/P_NL)(k_j, z_s) / R_EE(k_j, z_s)

R_X[s, j] is the base column (the response to P_X^truth on the ln k tent j
at snapshot s), R_dE, R_EE the truth's ratios to P_NL (P_X^truth = R_X P_NL),
so each term is the response to the model's own P_X on the tents. The same
sum with -C1_s/R_dE and C1_s^2/R_EE is the native NLA at A = 1, eta = 0
(written as N_GI, N_II, for the check against the emulator pipeline in
fisher_minimize.py --check-hm).

The templates are evaluated at the D3A cosmology (CAMB, Takahashi halofit,
the cosmology of the base); I_1 is the halo-model integral of
v6/halo_central_tables.npz. They are exact in the IA amplitudes and held
fixed in cosmology: at other cosmologies the alpha correction responds not
at all, while the native terms respond fully. The 1-halo and 2-halo
windows of the FLAMINGO fits (f_1h, f_2h) are not included; f_2h is above
0.97 below k = 1 h/Mpc.

Output: hm_templates_<probe>_<truth>.npz (probe cs: the 1080 xi_+- rows of
v9/ia_base_<truth>.npz; 3x2: the 2115 rows of ia_base_3x2_<truth>.npz,
gamma_t at the biases B1_FID), with zs, U_GI, U_II, N_GI, N_II and the
ratios used. Run with the cocoa Python (CAMB):
    ~/cocoa/Cocoa/.local/bin/python hm_templates.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "roman_dchi2")
D3A = dict(As=2.099e-9, ns=0.967, H0=68.1, omegab=0.0486, omegam=0.306, mnu=0.06, tau=0.0697186)


def camb_ratios(zs, k):
    """P_lin/P_NL at the D3A cosmology, as the v9 base computed P_NL."""
    import camb
    c = D3A
    h = c["H0"]/100
    omnuh2 = c["mnu"]*(3.046/3)**0.75/94.0708
    pars = camb.set_params(H0=c["H0"], ombh2=c["omegab"]*h*h, omch2=(c["omegam"] - c["omegab"])*h*h - omnuh2,
                           mnu=c["mnu"], tau=c["tau"], As=c["As"], ns=c["ns"], dark_energy_model="ppf",
                           halofit_version="takahashi")
    pars.Transfer.accurate_massive_neutrino_transfers = False
    pars.set_matter_power(redshifts=sorted(set(float(z) for z in zs), reverse=True), kmax=50.0,
                          k_per_logint=15, nonlinear=True)
    pars.NonLinear = camb.model.NonLinear_both
    res = camb.get_results(pars)
    kw = dict(hubble_units=False, k_hunit=False, extrap_kmax=300.0, var1="delta_tot", var2="delta_tot")
    PL = res.get_matter_power_interpolator(nonlinear=False, **kw)
    PN = res.get_matter_power_interpolator(nonlinear=True, **kw)
    pl = np.array([PL.P(z, k*h)*h**3 for z in zs])
    pn = np.array([PN.P(z, k*h)*h**3 for z in zs])
    D = np.sqrt(np.array([PL.P(z, 1e-3*h)/PL.P(0.0, 1e-3*h) for z in zs]))   # growth, D(0) = 1
    return pl, pn, D


def main():
    st = np.load(os.path.join(RD, "v3", "stitched_truths.npz"))
    ht = np.load(os.path.join(RD, "v6", "halo_central_tables.npz"))
    zs = st["zs"]
    assert np.allclose(ht["zs"], zs)
    for probe, base in (("cs", "ia_base"), ("3x2", "ia_base_3x2")):
        for truth in ("prod", "S1"):
            b = np.load(os.path.join(RD, "v9", f"{base}_{truth}.npz"))
            fine = b["fine"]
            assert np.allclose(b["zs"], zs)
            pl, pn, D = camb_ratios(zs, fine)
            # cosmolike's C1 bar = 0.01389 Omega_m/D(z) (the C1 of stitched_truths.npz
            # carries the notes' fitted amplitude and is not used)
            C1 = 0.01389*D3A["omegam"]/D
            # the halo tables' P_lin (same cosmology) as a check of the CAMB call
            plt = np.array([np.exp(np.interp(np.log(fine), np.log(ht["k_all"]), np.log(ht["Plin"][s])))
                            for s in range(len(zs))])
            i1 = np.array([np.interp(np.log(fine), np.log(ht["k_all"]), ht["I1m"][s]) for s in range(len(zs))])
            R = st[f"R_{truth}"]
            RdE = np.array([np.interp(np.log(fine), np.log(st["k"]), R[s, 0]) for s in range(len(zs))])
            REE = np.array([np.interp(np.log(fine), np.log(st["k"]), R[s, 1]) for s in range(len(zs))])
            assert np.all(RdE < 0) and np.all(REE > 0), "truth ratio changes sign: division unsafe"
            fgi = C1[:, None]*(1 - i1*pl/pn)/RdE           # dP_GI/P_GI^truth per unit (1-alpha) A_s
            fii = -C1[:, None]**2*(1 - pl/pn)/REE          # dP_II/P_II^truth per unit (1-alpha) A_s^2
            ngi, nii = -C1[:, None]/RdE, C1[:, None]**2/REE  # native NLA at A = 1
            U_GI = np.einsum("sj,sjn->sn", fgi, b["R_GI"])
            U_II = np.einsum("sj,sjn->sn", fii, b["R_II"])
            N_GI = np.einsum("sj,sjn->sn", ngi, b["R_GI"])
            N_II = np.einsum("sj,sjn->sn", nii, b["R_II"])
            out = os.path.join(HERE, f"hm_templates_{probe}_{truth}.npz")
            np.savez(out, zs=zs, fine=fine, U_GI=U_GI, U_II=U_II, N_GI=N_GI, N_II=N_II,
                     one_minus_I1Plin_over_PNL=1 - i1*pl/pn, one_minus_Plin_over_PNL=1 - pl/pn, I1=i1,
                     about=__doc__)
            j = [np.argmin(abs(fine - k)) for k in (0.1, 0.3, 1.0)]
            print(f"{probe} {truth}: P_lin CAMB vs halo tables max |dlnP| {np.max(abs(np.log(pl/plt))):.1e}; "
                  f"z=1 (1 - I1 Plin/PNL) at k 0.1, 0.3, 1: {np.round((1 - i1*pl/pn)[8, j], 3)}; "
                  f"|U_GI|/|N_GI| summed {np.abs(U_GI.sum(0)).max()/np.abs(N_GI.sum(0)).max():.3f}")


if __name__ == "__main__":
    main()
