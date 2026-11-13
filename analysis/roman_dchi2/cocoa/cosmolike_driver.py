"""Drive the compiled cosmolike_roman_real_interface (cosmolike core with the
sim-IA patch) without Cobaya, reproducing the roman_real likelihood's setup
(likelihood/_cosmolike_prototype_base.py: initialize, set_cosmo_related,
set_source_related) for the cosmic-shear probe.

Cosmology: CAMB with the settings of EXAMPLE_EVALUATE2_d3a_takahashi.yaml
(FLAMINGO D3A by default; halofit takahashi, AccuracyBoost 1.1, ppf dark
energy, kmax 10/Mpc with log extrapolation to 250/Mpc as Cobaya's
interpolator does). Grids (z_interp_1D, z_interp_2D, log10k_interp_2D) and
the growth normalisation follow set_cosmo_related exactly.

Usage:
    cl = Cosmolike(so_dir, data_dir)              # builds CAMB + cosmolike
    xi = cl.xi_native(ia_model=1, A1=[3.8, 0.57], A2=[-1.08, -0.71], BTA=[0.18])
    cl.use_sim_tables(table_dir, sn, zs, nfold)   # IA_code = 3
    xi = cl.xi_sim()
xi is [xi_+ (36 pairs x 15 theta) | xi_- (same)], the data-vector ordering
(validate_cosmolike.py: reproduces example1.modelvector to 3e-5 median).
"""
import os
import sys

import numpy as np

D3A = dict(As=2.099e-9, ns=0.967, H0=68.1, omegab=0.0486, omegam=0.306, mnu=0.06,
           tau=0.0697186, w=-1.0, wa=0.0)


class Cosmolike:
    def __init__(self, so_dir, data_dir, cosmo=D3A, accuracyboost=1.0, integration_accuracy=0,
                 lmax=100000, nz_file="example1.nz", ntheta=15, tmin=2.5, tmax=250.0,
                 threads=2, verbose=False, camb_boost=1.1, camb_kmax=10.0):
        self.camb_boost, self.camb_kmax = camb_boost, camb_kmax
        sys.path.insert(0, so_dir)
        import cosmolike_roman_real_interface as ci
        self.ci = ci
        self.cosmo = dict(cosmo)
        self.boost = accuracyboost
        self._grids()
        self._camb()
        ci.initial_setup()
        self._probes(ci)
        ci.init_binning(int(ntheta), tmin, tmax)
        ci.init_ggl_exclude(np.array([[6, 0], [7, 0], [7, 1]]).flatten())
        ci.set_log_level_info() if verbose else None
        if hasattr(ci, "init_photoz_histogram_akima"):   # removed in cocoa v5.x
            ci.init_photoz_histogram_akima(enabled=0, preparation_factor=3)
        ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
        ci.init_fpt_internal_boost(internal_boost=1.0)
        ci.init_nonlimber_accuracy_boost(nonlimber_boost=1.0)
        ci.init_adopt_limber_gs(adopt_limber_gs=0)
        ci.init_adopt_limber_gg(adopt_limber_gg=0)
        ci.init_include_HOD_GX(include_HOD_GX=0)
        ci.init_include_halo_IA(include_halo_IA=0)
        ci.init_ntable_lmax(lmax=int(lmax))
        ci.init_accuracy_boost(accuracy_boost=accuracyboost,
                               integration_accuracy=int(integration_accuracy))
        ci.init_cosmo_runmode(is_linear=False)
        nz = os.path.join(data_dir, nz_file)
        ci.init_redshift_distributions_from_files(lens_multihisto_file=nz, lens_ntomo=8,
                                                  source_multihisto_file=nz, source_ntomo=8)
        self._data(ci, data_dir, so_dir)
        ci.set_omp_threads(threads) if hasattr(ci, "set_omp_threads") else None
        self.sim = False

    def _probes(self, ci):
        ci.init_probes(possible_probes="xi")

    def _data(self, ci, data_dir, so_dir):
        # cosmolike needs a covariance to initialise its data block; only
        # compute_chi2 reads it. A diagonal stand-in (3-column format).
        dv = os.path.join(data_dir, "example1.modelvector")
        n = len(np.loadtxt(dv))
        covf = os.path.join(so_dir, "_diag_cov.txt")
        if not os.path.exists(covf):
            np.savetxt(covf, np.column_stack([np.arange(n), np.arange(n), np.ones(n)]),
                       fmt=["%d", "%d", "%.1f"])
        ci.init_data_real(covf, os.path.join(data_dir, "example1.mask"), dv)

    # ------------------------------------------------------------------
    def _grids(self):
        b = self.boost
        tmp = int(1000 + 250*b)
        self.z1 = np.concatenate((np.linspace(0.0, 3.0, max(100, int(0.80*tmp)), endpoint=False),
                                  np.linspace(3.0, 50.1, max(100, int(0.40*tmp)), endpoint=False),
                                  np.linspace(1070, 1100, max(50, int(0.10*tmp)))))
        m = int(min(2**np.ceil(np.log2(max(1.0, b))), 16))
        self.z2 = np.concatenate((np.linspace(0, 3.0, 105*m, endpoint=False),
                                  np.linspace(3.0, 49.99, 34*m + 1)))
        self.z2_camb = np.concatenate((np.linspace(0, 3.0, 105, endpoint=False),
                                       np.linspace(3.0, 49.99, 35)))
        self.lk = np.linspace(-4.99, 2.0, int(1250 + 250*b))

    def _camb(self):
        import camb
        c = self.cosmo
        h = c["H0"]/100.0
        omnuh2 = c["mnu"]*(3.046/3)**0.75/94.0708
        omch2 = (c["omegam"] - c["omegab"])*h*h - omnuh2
        pars = camb.set_params(H0=c["H0"], ombh2=c["omegab"]*h*h, omch2=omch2, mnu=c["mnu"],
                               tau=c["tau"], As=c["As"], ns=c["ns"], w=c["w"], wa=c["wa"],
                               dark_energy_model="ppf", halofit_version="takahashi",
                               AccuracyBoost=self.camb_boost)
        pars.Transfer.accurate_massive_neutrino_transfers = False
        pars.set_matter_power(redshifts=list(self.z2_camb[::-1]), kmax=self.camb_kmax, k_per_logint=15,
                              nonlinear=True)
        pars.NonLinear = camb.model.NonLinear_both
        res = camb.get_results(pars)
        self.res, self.h = res, h
        self.omegan2 = float(pars.omnuh2)
        kw = dict(hubble_units=False, k_hunit=False, extrap_kmax=250.0*self.boost,
                  var1="delta_tot", var2="delta_tot")
        self.PKL = res.get_matter_power_interpolator(nonlinear=False, **kw)
        self.PKNL = res.get_matter_power_interpolator(nonlinear=True, **kw)
        kwcb = dict(kw, var1="delta_nonu", var2="delta_nonu")
        self.PKLcb = res.get_matter_power_interpolator(nonlinear=False, **kwcb)
        self.sigma8 = res.get_sigma8_0()

    def set_cosmology(self):
        ci, h = self.ci, self.h
        k = 10.0**self.lk
        lnPL = np.log(self.PKL.P(self.z2, k)).flatten(order="F") + np.log(h**3)
        lnPNL = np.log(self.PKNL.P(self.z2, k)).flatten(order="F") + np.log(h**3)
        lnPLcb = np.log(self.PKLcb.P(self.z2, k)).flatten(order="F") + np.log(h**3)
        zg = self.z1[self.z1 <= self.z2[-1]]
        gk = 0.05
        G = np.sqrt(self.PKL.P(zg, gk)/self.PKL.P(0, gk))*(1 + zg)
        zn = self.z2[-1]
        G /= np.sqrt(self.PKL.P(zn, gk)/self.PKL.P(0, gk))*(1 + zn)
        G = np.asarray(G).ravel()
        ci.set_cosmology(omegam=self.cosmo["omegam"], omegab=self.cosmo["omegab"],
                         omegan2=self.omegan2, H0=self.cosmo["H0"],
                         log10k_2D=self.lk - np.log10(h), z_2D=self.z2,
                         lnP_linear=lnPL, lnP_linear_cb=lnPLcb, lnP_nonlinear=lnPNL,
                         G=G, z_G=zg, z_1D=self.z1,
                         chi=self.res.comoving_radial_distance(self.z1)*h)

    def _nuisance(self, A1, A2, BTA):
        ci = self.ci
        ci.set_nuisance_shear_calib(M=[0.0]*8)
        ci.set_nuisance_shear_photoz(bias=[0.0]*8)
        pad = lambda v: list(v) + [0.0]*(8 - len(v))
        ci.set_nuisance_ia(A1=pad(A1), A2=pad(A2), B_TA=pad(BTA))

    # ------------------------------------------------------------------
    def xi_native(self, ia_model=0, A1=(0.0,), A2=(0.0,), BTA=(0.0,), evolution=3):
        """cosmolike's own NLA (0) / TATT (1), C FAST-PT, power-law evolution."""
        self.ci.init_IA(ia_model=int(ia_model), ia_redshift_evolution=int(evolution), ia_code=0)
        self.sim = False
        self.set_cosmology()
        self._nuisance(A1, A2, BTA)
        return self.xi_vector()

    def use_sim_tables(self, table_dir, sn, zs, nfold=1):
        """IA_code = 3 with tables already in shear units at mu = 0:
        no shear conversion, no line-of-sight factor, no window correction."""
        ci = self.ci
        ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
        ci.init_IA_sim(str(table_dir), [int(s) for s in sn], [float(z) for z in zs], int(nfold))
        ci.set_IA_sim_shape_convention(0, 0)
        ci.set_IA_sim_window(0, 681.0, 1024)
        self.sim = True

    def xi_sim(self):
        self.set_cosmology()
        self._nuisance([0.0], [0.0], [0.0])
        return self.xi_vector()

    PAIRS = [(i, j) for i in range(8) for j in range(i, 8)]

    def xi_vector(self):
        """xi_pm_tomo returns two (Ntheta, 8, 8) arrays; reorder to the data
        vector layout [xi_+ (pairs i<=j, i-major, theta fastest) | xi_-],
        validated against example1.modelvector (median ratio 1.00003)."""
        xp, xm = (np.asarray(a) for a in self.ci.xi_pm_tomo())
        return np.concatenate([np.concatenate([a[:, i, j] for i, j in self.PAIRS])
                               for a in (xp, xm)])
