"""How well does an N-particle unreduced inertia tensor recover a galaxy's projected shape?

Sample N stars from a 3D ellipsoidal Hernquist-like profile with axis ratios (1, q, s), random
orientation, project along z, compute e = (Ixx-Iyy)/(Ixx+Iyy), 2Ixy/(...) about the true centre
(the SOAP/TNG unreduced tensor of all bound stars). Compare to the N -> infinity value.

Reported per N:
  response  R = <e_N . e_true> / <|e_true|^2>   (the factor that multiplies a P_hE alignment signal)
  sigma_e   per-component rms of e_N - e_true    (extra shape noise)
"""
import json
import os

import numpy as np

rng = np.random.default_rng(1)


def sample(n, q, s, rot):
    # Hernquist radial profile, r = a * sqrt(u)/(1 - sqrt(u)); truncate at 20 a (bound-star cut)
    u = rng.uniform(0, 1, n)
    r = np.sqrt(u) / (1 - np.sqrt(u))
    r = np.minimum(r, 20.0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    x = r[:, None] * v * np.array([1.0, q, s])
    return x @ rot.T


def ell(x):
    ixx, iyy, ixy = np.mean(x[:, 0] ** 2), np.mean(x[:, 1] ** 2), np.mean(x[:, 0] * x[:, 1])
    return np.array([(ixx - iyy) / (ixx + iyy), 2 * ixy / (ixx + iyy)])


def rand_rot():
    a = rng.normal(size=(3, 3))
    qm, r = np.linalg.qr(a)
    return qm * np.sign(np.diag(r))


NGAL = 4000
rows = []
print(f"{'N':>6s} {'response R':>11s} {'sigma_e(extra)':>15s} {'e_rms(N)':>9s} {'e_rms(true)':>11s}")
for n in [10, 13, 20, 30, 50, 100, 300, 1000]:
    et, en = [], []
    for _ in range(NGAL):
        q, s = rng.uniform(0.5, 0.95), rng.uniform(0.3, 0.6)
        s = min(s, q)
        rot = rand_rot()
        # "true" shape: analytic second moments of the ellipsoid (isotropic profile * axes)
        c = rot @ np.diag([1.0, q * q, s * s]) @ rot.T
        et.append([(c[0, 0] - c[1, 1]) / (c[0, 0] + c[1, 1]), 2 * c[0, 1] / (c[0, 0] + c[1, 1])])
        en.append(ell(sample(n, q, s, rot)))
    et, en = np.array(et), np.array(en)
    R = np.sum(en * et) / np.sum(et * et)
    sig = np.std(en - R * et)
    rows.append({"N": n, "response": float(R), "sigma_extra": float(sig), "erms_N": float(np.sqrt(np.mean(en ** 2))),
                 "erms_true": float(np.sqrt(np.mean(et ** 2)))})
    print(f"{n:6d} {R:11.3f} {sig:15.3f} {np.sqrt(np.mean(en ** 2)):9.3f} {np.sqrt(np.mean(et ** 2)):11.3f}")
json.dump({"about": "discreteness_mc.py, seed 1, 4000 galaxies per N", "rows": rows},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "discreteness_mc.json"), "w"), indent=1)
