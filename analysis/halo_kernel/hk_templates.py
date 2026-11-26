#!/usr/bin/env python3
"""Roman templates of the kernel-calibrated halo model (HMk) for fisher_minimize.py.

The HMk prediction at snapshot s (hk_model.py, one variant) gives P_hE, P_EE,
P_BB in the cosmolike-fit convention. As for the CP truth, P_dE = P_hE/b(k)
with FLAMINGO's b(k), and the ratios to P_NL are
    R_dE = P_dE/P_NL,  R_EE = P_EE/P_NL,  R_BB = P_BB/P_NL
(the 1-halo II term, the close-pair term, is part of R_EE and R_BB). Its
amplitude A_eff,s = A_truth,s <R_dE/R_dE^truth> over KNORM (A_truth,s = <R_dE^truth/(-C1_s)>
over KA, the truth's linear limit) defines the shape per unit amplitude. In Roman the model is
    native NLA(A1, eta1) + sum_s [A_s U_GI[s] + A_s^2 U_II[s]],
    A_s = A1 ((1+z_s)/1.62)^eta1,
    U_GI[s] = sum_j R_GI[s,j] [R_dE/A_eff + C1]/R_dE^truth (k_j)
    U_II[s] = sum_j R_II_E[s,j] [R_EE/A_eff^2 - C1^2]/R_EE^truth (k_j)
            + sum_j R_II_B[s,j] [R_BB/A_eff^2]/R_BB^truth (k_j)
(R_X[s,j] the response of the data vector to the truth's P_X on the ln k tent j).
so that at A_s = A_eff,s the data vector is the HMk prediction, with every
cosmology dependence of native NLA kept and the shape held at the fiducial
cosmology (as hm_templates.py). Outside the measured range (KLO to KHI) the
model's ratio to the truth, R_model/R_truth, is held at its edge value.

Writes U_GI_hmk, U_II_hmk, A_eff_hmk into cut_validation/hm_templates_<probe>_CP.npz
(model NLAk of fisher_minimize.py).
    python hk_templates.py [variant] [suffix]   (default w-ls-grad@1.26, suffix hmk)
A variant <name>-grad@<k_max> takes the kernel-gradient fit to k <= k_max at each
snapshot (hk_model.py stores one fit per k_max of KGRID).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hk_fields as HF  # noqa: E402
from hk_model import KGRID  # noqa: E402

RD = os.path.join(HF.REPO, 'analysis', 'roman_dchi2')
CV = os.path.join(HF.REPO, 'analysis', 'cut_validation')
OMEGA_M = 0.306
KA = (0.03, 0.10)       # large-scale range of the truth's amplitude [h/Mpc]
# the model is used on KLO <= k <= KHI: below KLO the single box's cosmic variance
# (data 4 to 19 percent above the smooth truth at k < 0.15) would set the template's
# amplitude, above KHI the model was not tested (and alpha k^2 keeps growing);
# outside, the model's ratio to the truth is held at its edge value
KLO, KHI = 0.2, 1.26
KNORM = (0.2, 0.4)      # the model's amplitude relative to the truth's is its GI ratio here


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'w-ls-grad@1.26'
    sfx = sys.argv[2] if len(sys.argv) > 2 else 'hmk'
    # optional third argument: use the model only at z <= zmax and the truth's own
    # shape above (a diagnostic of where the model's error enters Roman)
    zmax = float(sys.argv[3]) if len(sys.argv) > 3 else np.inf
    # optional fourth argument: the variant used above zmax (default: the truth's own shape)
    hivar = sys.argv[4] if len(sys.argv) > 4 else 'truth'
    res = json.load(open(os.path.join(HERE, 'model', 'hk_model.json')))
    fits = np.load(os.path.join(RD, 'v2', 'fits.npz'))
    cp = np.load(os.path.join(RD, 'v3', 'cp_truth.npz'))
    zs = cp['zs']
    sys.path.insert(0, os.path.join(CV))
    import hm_templates as HT                                   # noqa: E402
    for probe, base in (('cs', 'ia_base'), ('3x2', 'ia_base_3x2')):
        b = np.load(os.path.join(RD, 'v9', f'{base}_CP.npz'))
        fine = b['fine']
        _, _, D = HT.camb_ratios(zs, fine)
        C1 = 0.01389*OMEGA_M/D
        RdE = np.array([np.interp(np.log(fine), np.log(cp['k']), cp['R_CP'][s, 0]) for s in range(len(zs))])
        REE = np.array([np.interp(np.log(fine), np.log(cp['k']), cp['R_CP'][s, 1]) for s in range(len(zs))])
        RBB = np.array([np.interp(np.log(fine), np.log(cp['k']), cp['R_CP_B'][s, 1]) for s in range(len(zs))])
        UG = np.zeros((len(zs), b['R_GI'].shape[2]))
        UI = np.zeros_like(UG)
        Aeff = np.zeros(len(zs))
        for s, z in enumerate(zs):
            sn = int(fits['sns'][s])
            line = res.get(str(sn))
            kf = np.array(line['k']) if line else None
            v = variant if z <= zmax else hivar
            if v == 'truth':                         # matched control: the CP truth's own shape
                ok = (cp['k'] >= KLO) & (cp['k'] <= KHI)
                kf = cp['k']
                r_de, r_ee, r_bb = cp['R_CP'][s, 0][ok], cp['R_CP'][s, 1][ok], cp['R_CP_B'][s, 1][ok]
            elif '@' in v:
                name, km = v.split('@')
                jk = int(np.argmin(np.abs(np.array(KGRID) - float(km))))
                P = np.array(line[name]['pred_kmax'][jk])
            else:
                P = np.array(line[v]['pred'])
            if v != 'truth':
                ok = (kf >= KLO) & (kf <= KHI)
                pnl, bk = fits['PNL'][s], fits['b'][s]
                r_de = (P[0]/bk/pnl)[ok]
                r_ee = (P[1]/pnl)[ok]
                r_bb = (P[2]/pnl)[ok]
            lk = np.log(kf[ok])
            kt = (cp['k'] >= KA[0]) & (cp['k'] <= KA[1])
            A_truth = np.mean(cp['R_CP'][s, 0][kt]/(-C1[s]))
            # the truth on the model's k values: the model enters as its ratio to the
            # truth, f = R_model/R_truth, held constant outside the model's k range
            # (holding R_model itself constant would misplace the truth's own shape
            # beyond k = 3.1, chi2 = 1.7 at K0 = 0.3 for the truth's own template)
            tr = [np.interp(lk, np.log(cp['k']), cp[key][s, c]) for key, c in (('R_CP', 0), ('R_CP', 1), ('R_CP_B', 1))]

            kn = (kf[ok] >= KNORM[0]) & (kf[ok] <= KNORM[1])
            Aeff[s] = A_truth*np.mean(r_de[kn]/tr[0][kn])

            def onfine(r):
                return np.interp(np.log(fine), lk, r)
            fgi = onfine(r_de/tr[0])/Aeff[s] + C1[s]/RdE[s]
            fee = onfine(r_ee/tr[1])/Aeff[s]**2 - C1[s]**2/REE[s]
            fbb = onfine(r_bb/tr[2])/Aeff[s]**2
            UG[s] = np.einsum('j,jn->n', fgi, b['R_GI'][s])
            UI[s] = np.einsum('j,jn->n', fee, b['R_II_E'][s]) + np.einsum('j,jn->n', fbb, b['R_II_B'][s])
        fn = os.path.join(CV, f'hm_templates_{probe}_CP.npz')
        t = dict(np.load(fn))
        t.update({f'U_GI_{sfx}': UG, f'U_II_{sfx}': UI, f'A_eff_{sfx}': Aeff, f'{sfx}_variant': f'{variant} (z <= {zmax}), {hivar} above'})
        np.savez(fn, **t)
        print(f'{probe} ({sfx} = {variant}): A_eff per snapshot {np.round(Aeff, 2)}; |U_GI|/|N_GI| summed '
              f'{np.abs(UG.sum(0)).max()/np.abs(t["N_GI"].sum(0)).max():.3f}; wrote {fn}', flush=True)


if __name__ == '__main__':
    main()
