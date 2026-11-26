#!/usr/bin/env python3
"""Shape-shape spectra of the model galaxy shape field, split into the part of
the halo shapes correlated with the halo density and the rest.

Why: the final model (w-ls in hk_model.py) predicts the 2-halo II from the
deterministic halo shapes only, (sum_j kappa_j P_{H,E_j})^2 / P_HH, and sets
the 2-halo BB to zero. At z <= 1.5 this matches FLAMINGO; at z >= 2 the
measured II and BB are 15 to 45 percent above it at k = 0.2 to 0.5.

Decomposition. Each halo's projected DM shape is the sum of a part predicted
by the halo density field H around it (the deterministic part, linear in
delta_H in Fourier space) and a residual: e_h = e_h^det + e_h^res. A galaxy
can inherit the two with different strengths,
    e_g = kappa_d e_h^det + kappa_r e_h^res + (random),
  kappa_d  is measured by the large-scale cross with the galaxy density
           (kappa_w in hk_model.py: only e^det correlates with density there),
  kappa_r  is measured by the object-by-object regression of e_g on e_h
           (kappa_c, kappa_s of hk_measure.py: e^res carries almost all of the
           variance of a halo's shape, so the regression slope is kappa_r).
The model galaxy shape field is then E_g = E_r + (T_d - T_r)/P_HH delta_H with
E_x = sum_j kappa_x,j E_j (the mean-HOD-weighted halo shape fields of
hk_weighted.py) and T_x = P_{H, E_x}, and
    P_EE^2h = T_d^2 / P_HH + [P_{E_r E_r} - T_r^2 / P_HH],
    P_BB^2h = P_{B_r B_r},
the deterministic II of w-ls plus the residual halo-halo shape correlation
weighted by the object-level kernels. No parameter is fitted.

Fields (512 mesh, folds 1 and 4, rescaled to the pipeline's 1024 mesh as
hk_weighted.py): H, G as hk_weighted; Ed = sum_j kappa_d,j (Ec_j | Es_j) and
Er = sum_j kappa_r,j (Ec_j | Es_j). The white noise of the shape autos (each
halo with itself, including a central with its own halo's satellites) is
measured by repeating the shape fields with every halo shape rotated by a
random angle and subtracted before the mesh rescaling.

Spectra stored (raw multipoles [3, nk], noise off): Ed x Ed (E, B),
Er x Er (E, B), Ed x Er (E), H x Ed, H x Er, G x Ed, G x Er (E), H x H.
Writes kernels/stoch_<sn>.npz, with the kernels used. About 5 minutes per snapshot.
    python hk_stoch.py <sn> [<sn> ...]      (after hk_model.py has written kappa_w)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hk_fields as HF  # noqa: E402
import hk_measure as HM  # noqa: E402
import hk_weighted as HW  # noqa: E402

NC_MIN, N_MIN = 0.01, 1000      # as hk_model.py: kernels from too few galaxies are set to zero


def kernels(sn, ker):
    '''kappa_d (large-scale, from hk_model.json) and kappa_r (object regression).'''
    res = json.load(open(os.path.join(HERE, 'model', 'hk_model.json')))[str(sn)]
    kd = (np.array(res['kappa_w_c']), np.array(res['kappa_w_s']))
    kr_c = np.nan_to_num(ker['kappa_c'].copy())
    kr_s = np.nan_to_num(ker['kappa_s'].copy())
    nobj_c = ker['n_h']*HF.BOX**3*ker['Nc']
    nobj_s = ker['n_h']*HF.BOX**3*ker['Ns']
    kr_c[(ker['Nc'] < NC_MIN) | (nobj_c < N_MIN)] = 0.0
    kr_s[nobj_s < N_MIN] = 0.0
    return kd, (kr_c, kr_s)


def composite(h, e, w_c, w_s, kap, nG_V, nf, us, kus):
    '''Shape grids sum_j kappa_c,j Ec_j + kappa_s,j Es_j (Es filtered by u_s,j).'''
    out = None
    for j in range(HM.NB):
        inb = h['bin'] == j
        for kv, w, u in ((kap[0][j], w_c, None), (kap[1][j], w_s, us[j])):
            if kv == 0 or w[inb].sum() <= 0:
                continue
            sh = HW.shape_field(h['centre'][inb], kv*w[inb], e[inb], nG_V, nf, u=u, kus=kus)
            if out is None:
                out = sh
            else:
                for c in sh:
                    out[c] += sh[c]
            del sh
    return out


def one_fold(g, h, ker, kd, kr, nf, t0, rng):
    vol = HF.BOX**3
    nc, ns, wc, ws = h['nc'], h['ns'], h['wc'], h['ws']
    kus, us = ker['k'], ker['us_k']
    okk = np.isfinite(kus) & (kus > 0)
    kus, us = kus[okk], us[:, okk]
    nG = (wc.sum() + ws.sum())/vol
    parts = []
    for i in range(HM.NB):
        inb = h['bin'] == i
        for w, ub in ((wc, None), (ws, i)):
            parts.append((h['centre'][inb], w[inb], w[inb].sum()/vol/nG, ub))
    hm = h['lgm'] >= HW.M_COMPLETE
    fields = {'G': {'d': HW.combo_density(parts, nf, us, kus)},
              'H': HF.tracer_fields(h['centre'][hm], None, nfold=nf)}
    okh = np.all(np.isfinite(h['e']), axis=1)
    e = np.where(okh[:, None], h['e'], 0.0)
    wc0, ws0 = np.where(okh, wc, 0.0), np.where(okh, ws, 0.0)
    fields['Ed'] = composite(h, e, wc0, ws0, kd, nG*vol, nf, us, kus)
    fields['Er'] = composite(h, e, wc0, ws0, kr, nG*vol, nf, us, kus)
    pairs = [('Ed', 'E', 'Ed', 'E'), ('Ed', 'B', 'Ed', 'B'), ('Er', 'E', 'Er', 'E'), ('Er', 'B', 'Er', 'B'),
             ('Ed', 'E', 'Er', 'E'), ('H', 'd', 'Ed', 'E'), ('H', 'd', 'Er', 'E'), ('G', 'd', 'Ed', 'E'),
             ('G', 'd', 'Er', 'E'), ('H', 'd', 'H', 'd')]
    k, nm, P = HF.power(fields, pairs, nfold=nf)
    print(f'    fold {nf}: fields and spectra done [{time.time() - t0:.0f} s]', flush=True)
    for name in ('Ed', 'Er', 'G'):
        del fields[name]
    # noise: the same shape fields with every halo shape rotated by a random angle
    phi = rng.uniform(0, np.pi, len(e))
    c2, s2 = np.cos(2*phi), np.sin(2*phi)
    erot = np.column_stack([e[:, 0]*c2 - e[:, 1]*s2, e[:, 0]*s2 + e[:, 1]*c2])
    rot = {'Ed': composite(h, erot, wc0, ws0, kd, nG*vol, nf, us, kus),
           'Er': composite(h, erot, wc0, ws0, kr, nG*vol, nf, us, kus)}
    auto = [p for p in pairs if p[0] in ('Ed', 'Er') and p[2] in ('Ed', 'Er')]
    _, _, PN = HF.power(rot, auto, nfold=nf)
    del rot
    for p in auto:
        P[p] = P[p] - PN[p]
    P[('H', 'd', 'H', 'd')][0] -= vol/hm.sum()
    print(f'    fold {nf}: noise spectra done [{time.time() - t0:.0f} s]', flush=True)
    return k, nm, HF.to_pipeline_mesh(k, P, nf)


def main():
    rng = np.random.default_rng(20261126)
    for s in sys.argv[1:]:
        sn = int(s)
        t0 = time.time()
        g, h = HM.load(sn)
        ker = dict(np.load(os.path.join(HM.OUT, f'kernels_{sn}.npz')))
        nc, ns, wc, ws = HW.mean_hod(h, g)
        h.update(nc=nc, ns=ns, wc=wc, ws=ws)
        kd, kr = kernels(sn, ker)
        print(f'sn {sn} (z = {g["z"]:.2f}): kappa_d c {np.round(kd[0], 2)} s {np.round(kd[1], 2)}; '
              f'kappa_r c {np.round(kr[0], 2)} s {np.round(kr[1], 2)}', flush=True)
        per = {}
        for nf, lo, hi in HF.FOLDS:
            per[nf] = one_fold(g, h, ker, kd, kr, nf, t0, rng)
        k, nm, P = HF.stitch(per)
        np.savez(os.path.join(HM.OUT, f'stoch_{sn}.npz'), k=k, nmodes=nm, z=g['z'],
                 kappa_d_c=kd[0], kappa_d_s=kd[1], kappa_r_c=kr[0], kappa_r_s=kr[1],
                 **{'S|' + '|'.join(key): v for key, v in P.items()})
        print(f'  wrote stoch_{sn}.npz [{time.time() - t0:.0f} s]', flush=True)


if __name__ == '__main__':
    main()
