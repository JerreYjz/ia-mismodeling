"""B-mode II for the base builders (ia_base.py, ia_base_3x2.py).

A truth with a B-mode II table R_<truth>_B (v3/cp_truth.npz) adds C_BB with
the same Limber kernel as C_EE. The sim_IA path computes E modes only, and
xi_+- = int (C_EE +- C_BB), so the B part of the data vector is (xi_+, -xi_-)
of the same table run as E, and nothing in gamma_t or w (v9/bmode_truth_3x2.py).

    st = load_truths(stitched_truths_path)   # S1, prod, ... plus CP, CP_B, ZERO
    flip_b(v)                                 # E-run vector -> its B-mode vector
    EBModel(mE, mB, mZ)                       # truth-as-model with the B part
"""
import os

import numpy as np

NXI = 1080                       # xi_+ then xi_-, 540 each
HERE = os.path.dirname(os.path.abspath(__file__))
CP_FILE = os.path.join(HERE, "..", "v3", "cp_truth.npz")


def load_truths(path):
    """stitched_truths.npz as a dict, with the CP tables and an all-zero table."""
    st = dict(np.load(path))
    if os.path.exists(CP_FILE):
        cp = np.load(CP_FILE)
        assert np.allclose(cp["k"], st["k"]) and np.allclose(cp["zs"], st["zs"])
        st["R_CP"], st["R_CP_B"] = cp["R_CP"], cp["R_CP_B"]
    st["R_ZERO"] = np.zeros_like(st["R_S1"])
    return st


def has_b(st, truth):
    return f"R_{truth}_B" in st


def flip_b(v):
    """The B-mode vector of an E run: xi_+ kept, xi_- negated, the rest zero."""
    out = np.zeros_like(v)
    out[:NXI//2] = v[:NXI//2]
    out[NXI//2:NXI] = -v[NXI//2:NXI]
    if len(v) > NXI:
        assert np.max(np.abs(v[NXI:])) <= 1e-12*np.max(np.abs(v[:NXI])) + 1e-30, "II leaked into gamma_t or w"
    return out


class EBModel:
    """Truth-as-model with B-mode II: E model + flip_b(B model - zero model).

    The three models share cosmology, (A, eta) (B scales as II, with A^2) and
    biases; the zero-table model removes the GG (and gamma_t, w) part of the B
    run, so only the B-mode II enters."""

    def __init__(self, mE, mB, mZ):
        self.mE, self.mB, self.mZ = mE, mB, mZ

    def __call__(self, p, ia):
        return self.mE(p, ia) + flip_b(self.mB(p, ia) - self.mZ(p, ia))
