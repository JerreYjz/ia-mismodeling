#!/usr/bin/env python3
"""cosmolike 3x2pt driver (xi_+-, gamma_t, w) built on cosmolike_driver.Cosmolike.

Data vector layout (cocoa_roman_real, 15 theta bins 2.5-250'):
  [0, 540) xi_+ (36 source pairs i<=j) | [540, 1080) xi_- | [1080, 1995) gamma_t
  (61 lens-source pairs, ggl_exclude [6,0],[7,0],[7,1]) | [1995, 2115) w (8 lens bins)
Lens and source samples share example1.nz (as the dataset does). Linear
galaxy bias b_i (B1_FID of the wrappers), no magnification, no point mass.
Limber for gs (required by the sim_IA path) and gg (speed), for truth and
models alike. IA enters gamma_t through the GI term b_i P_dE; the sim_IA
tables carry the FLAMINGO P_dE there exactly as in xi_+-.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cosmolike_driver import Cosmolike, D3A  # noqa: E402

B1_FID = [1.24, 1.36, 1.55, 1.71, 1.90, 2.15, 2.52, 2.44]
NDV = 2115


class Cosmolike3x2(Cosmolike):
    def __init__(self, so_dir, data_dir, cosmo=D3A, **kw):
        self._data_dir = data_dir
        super().__init__(so_dir, data_dir, cosmo=cosmo, **kw)

    def _probes(self, ci):
        ci.init_probes(possible_probes="3x2pt")

    def _data(self, ci, data_dir, so_dir):
        dv = os.path.join(data_dir, "example1.modelvector")
        n = len(np.loadtxt(dv))
        assert n == NDV, n
        covf = os.path.join(so_dir, "_diag_cov3x2.txt")
        if not os.path.exists(covf):
            np.savetxt(covf, np.column_stack([np.arange(n), np.arange(n), np.ones(n)]), fmt=["%d", "%d", "%.1f"])
        ci.init_data_real(covf, os.path.join(data_dir, "ones.mask"), dv)
        ci.init_bias(bias_model=[0, 0, 0, 1, 0, 0])
        ci.init_adopt_limber_gs(adopt_limber_gs=1)
        ci.init_adopt_limber_gg(adopt_limber_gg=1)

    def _nuisance(self, A1, A2, BTA, B1=None):
        ci = self.ci
        pad = lambda v: list(v) + [0.0]*(8 - len(v))
        ci.set_nuisance_shear_calib(M=[0.0]*8)
        ci.set_nuisance_shear_photoz(bias=[0.0]*8)
        ci.set_nuisance_clustering_photoz(bias=[0.0]*8)
        ci.set_nuisance_ia(A1=pad(A1), A2=pad(A2), B_TA=pad(BTA))
        b1 = list(B1_FID if B1 is None else B1)
        ci.set_nuisance_bias(B1=b1, B2=[0.0]*8, B_MAG=[0.0]*8, B3nl=[0.0]*8, BK=[0.0]*8)
        ci.set_point_mass(PMV=[0.0]*8)
        ci.reset_bary_struct()

    def dv_native(self, ia_model=0, A1=(0.0,), A2=(0.0,), BTA=(0.0,), B1=None, evolution=3):
        self.ci.init_IA(ia_model=int(ia_model), ia_redshift_evolution=int(evolution), ia_code=0)
        self.sim = False
        self.set_cosmology()
        self._nuisance(A1, A2, BTA, B1)
        return np.asarray(self.ci.compute_data_vector_masked(), dtype=float)

    def dv_sim(self, B1=None):
        """Full 3x2pt vector with the sim_IA tables loaded by use_sim_tables."""
        self.set_cosmology()
        self._nuisance([0.0], [0.0], [0.0], B1)
        return np.asarray(self.ci.compute_data_vector_masked(), dtype=float)

    def gs_pairs(self):
        v = np.asarray(self.ci.get_gs_redshift_bins()).reshape(-1, 2)
        return [(int(a), int(b)) for a, b in v]
