#!/usr/bin/env python3
"""beta_NL integrals at the 13 snapshots of the Roman IA base (z = 0 to 4),
for the Roman templates of cut_validation/hm_templates.py.

beta_tables.py and halo_nlbias.py make the same tables at the nine core
FLAMINGO snapshots (z = 0.2 to 2). The Roman base (v9/ia_base_*.npz,
v6/halo_strict_tables.npz) also has z = 0, 0.1, 3 and 4. This script runs the
same code at all 13:
  * beta_NL from Dark Emulator (beta_tables.beta_at) for z <= 1.476; above,
    the z = 1.476 table assigned to the masses of the same peak height
    (as beta_tables.py does for z = 1.5 and 2; at z = 3 and 4 this is a
    stronger extrapolation, and their weight in Roman's lensing kernels is
    small);
  * the integrals I^beta_cm, I^beta_cc (halo_nlbias.tables) with the HOD of
    halo_strict_tables.npz per snapshot, shape bias proportional to b(M),
    beta_NL held at the nearest emulator bin below its lowest mass.
The nine core snapshots reproduce nlbias_tables.npz (checked below).

Output nlbias_roman_tables.npz: zs [13], k (= k_all of v6, 350 points),
I1, Ib_cm, Ib_cc, Ib_cm_lo, Ib_cc_lo [13, nk], lgM_E [13, 9].
Run (Dark Emulator venv, CAMB from the cocoa env), about 30 seconds:
    PYTHONPATH=~/cocoa/Cocoa/.local/lib/python3.11/site-packages \
        ~/cocoa/venvs/darkemu/bin/python nlbias_roman_tables.py
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beta_tables as BTB  # noqa: E402
import halo_nlbias as HN  # noqa: E402

H = BTB.H
V6 = BTB.V6


def main():
    from dark_emulator import darkemu
    t0 = time.time()
    st = dict(np.load(os.path.join(V6, 'halo_strict_tables.npz')))
    ct = dict(np.load(os.path.join(V6, 'halo_central_tables.npz')))
    zs = st['zs']
    k = ct['k_all']
    emu = darkemu.base_class()
    emu.set_cosmology(BTB.cosmology())
    lin = H.Linear(list(zs) + [BTB.Z_EMU])
    nz = len(zs)
    bt = {'k': BTB.K, 'beta': np.zeros((nz, BTB.NM_E, BTB.NM_E, BTB.K.size)), 'lgM_E': np.zeros((nz, BTB.NM_E))}
    lgm_fine = np.linspace(9.0, 16.0, 701)
    b_e, _, lg_e, _ = BTB.beta_at(emu, BTB.Z_EMU)
    for s, z in enumerate(zs):
        if z <= BTB.Z_EMU + 1e-9:
            bt['beta'][s], _, bt['lgM_E'][s], _ = BTB.beta_at(emu, float(z))
        else:
            bt['beta'][s] = b_e
            bt['lgM_E'][s] = np.interp(BTB.nu_of(lin, BTB.Z_EMU, lg_e), BTB.nu_of(lin, float(z), lgm_fine), lgm_fine)
    out = {q: np.zeros((nz, k.size)) for q in ('I1', 'Ib_cm', 'Ib_cc', 'Ib_cm_lo', 'Ib_cc_lo')}
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), k, nM=HN.NM, nx=10)
        lgMmin, lgM1 = st['hod'][s, :2]
        hm.set_hod(lgMmin, HN.SIG_LGM, lgM1, alpha=HN.ALPHA_SAT)
        assert abs(hm.ng/st['hod'][s, 3] - 1) < 1e-3, (z, hm.ng, st['hod'][s, 3])
        I0 = trapezoid(hm.dndlnM*hm.b*hm.M/H.RHO_CB, hm.lnM)
        lgM = hm.lnM/np.log(10)
        full = HN.tables(hm, HN.beta_on_grid(bt, s, lgM, k), I0)
        zero = HN.tables(hm, HN.beta_on_grid(bt, s, lgM, k, low='zero'), I0)
        out['I1'][s] = full['I1']
        out['Ib_cm'][s], out['Ib_cc'][s] = full['Ib_cm'], full['Ib_cc']
        out['Ib_cm_lo'][s], out['Ib_cc_lo'][s] = zero['Ib_cm'], zero['Ib_cc']
        assert np.allclose(full['I1'], ct['I1m'][s], rtol=1e-4), 'I_1 differs from v6'
        ik = [int(np.argmin(np.abs(k - kk))) for kk in (0.1, 0.3, 0.5, 1.0)]
        print(f'z {z:4.2f}: lgM_E {bt["lgM_E"][s, 0]:.2f} to {bt["lgM_E"][s, -1]:.2f}; at k 0.1, 0.3, 0.5, 1: '
              f'Ib_cm {np.array2string(full["Ib_cm"][ik], precision=3)}, Ib_cc {np.array2string(full["Ib_cc"][ik], precision=3)}, '
              f'Ib_cm_lo {np.array2string(zero["Ib_cm"][ik], precision=3)}  [{time.time() - t0:.0f} s]', flush=True)
    # the nine core snapshots must reproduce nlbias_tables.npz
    core = np.load(os.path.join(HERE, 'nlbias_tables.npz'))
    for t, z in enumerate(core['zs']):
        s = int(np.argmin(np.abs(zs - z)))
        for q in ('Ib_cm', 'Ib_cc', 'Ib_cm_lo'):
            d = np.max(np.abs(out[q][s] - core[q][t]))
            assert d < 1e-6, (z, q, d)
    print('core snapshots reproduce nlbias_tables.npz')
    np.savez(os.path.join(HERE, 'nlbias_roman_tables.npz'), zs=zs, k=k, lgM_E=bt['lgM_E'], **out)
    print('wrote nlbias_roman_tables.npz')


if __name__ == '__main__':
    main()
