#!/usr/bin/env python3
"""Dark Emulator beta_NL(k, M1, M2) at the 13 FLAMINGO snapshots of the IA base.

The same construction as analysis/halo_nlbias/beta_tables.py (mass bins
between cumulative-density thresholds, beta normalised to 0 at
k = 0.01 to 0.03, the z = 1.476 table assigned by peak height above the
emulator's last redshift), stored on its emulator mass grid lgM_E so that
hk_model.py can interpolate it to the halo-mass bins of the kernel model.

Output kernels/beta_emu.npz: zs [13], k, lgM_E [13, 9], beta [13, 9, 9, nk].
Run in the Dark Emulator environment, about 30 seconds:
    PYTHONPATH=~/cocoa/Cocoa/.local/lib/python3.11/site-packages \
        ~/cocoa/venvs/darkemu/bin/python hk_beta_emu.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'halo_nlbias'))
import beta_tables as BTB  # noqa: E402

ZS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def main():
    from dark_emulator import darkemu
    emu = darkemu.base_class()
    emu.set_cosmology(BTB.cosmology())
    lin = BTB.H.Linear(list(ZS) + [BTB.Z_EMU])
    beta = np.zeros((len(ZS), BTB.NM_E, BTB.NM_E, BTB.K.size))
    lgm = np.zeros((len(ZS), BTB.NM_E))
    b_e, _, lg_e, _ = BTB.beta_at(emu, BTB.Z_EMU)
    fine = np.linspace(9.0, 16.0, 701)
    for s, z in enumerate(ZS):
        if z <= BTB.Z_EMU + 1e-9:
            beta[s], _, lgm[s], _ = BTB.beta_at(emu, z)
        else:
            beta[s] = b_e
            lgm[s] = np.interp(BTB.nu_of(lin, BTB.Z_EMU, lg_e), BTB.nu_of(lin, z, fine), fine)
        print(f'z {z:4.2f}: lgM_E {lgm[s, 0]:.2f} to {lgm[s, -1]:.2f}', flush=True)
    os.makedirs(os.path.join(HERE, 'kernels'), exist_ok=True)
    np.savez(os.path.join(HERE, 'kernels', 'beta_emu.npz'), zs=np.array(ZS), k=BTB.K, lgM_E=lgm, beta=beta)
    print('wrote kernels/beta_emu.npz')


if __name__ == '__main__':
    main()
