#!/bin/bash
# End-to-end test on tiny mock SOAP catalogues (ng=64, a few seconds).
#   bash tests/run_mock_test.sh [workdir]
# Checks: masked tensors dropped, ellipticities match the tensors, and a
# radially aligned sample gives P_hE >> P_hB while a random one gives P_hE ~ 0.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
W="${1:-$(mktemp -d)}"
mkdir -p "$W"
cd "$W"

for al in 1 0; do
  python3 "$HERE/tests/make_mock_soap.py" "$W/soap_al$al.hdf5" $al 1
  for shape in BoundSubhalo/StellarInertiaTensorNoniterative \
               ProjectedAperture/50kpc/projz/ProjectedStellarInertiaTensorNoniterative; do
    tag=al${al}_$(echo "$shape" | cut -d/ -f1)
    cat > "$W/sg_$tag.dat" <<EOF
SoapFile      $W/soap_al$al.hdf5
OutDirBase    $W/$tag/cat/
SnapNum       78
M_min         0
M_max         1e4
ShapeField    $shape
EOF
    cat > "$W/gds_$tag.dat" <<EOF
GalCatDir     $W/$tag/cat/
OutDirBase    $W/$tag/gal/
SnapNum       78
Ngrid         64
EOF
    python3 "$HERE/soap_galaxies.py" "sg_$tag.dat" > "sg_$tag.log" 2>&1
    for nf in 1 2; do
      python3 "$HERE/galaxy_ds_field.py" "gds_$tag.dat" $nf > "gds_${tag}_$nf.log" 2>&1
      python3 "$HERE/ia_power.py" boxsize=204.3 kmin=0.03 kmax=3 nbin=8 munbin=4 ng=64 \
          sn=78 nfold=$nf base="$W/$tag/" gal_dir=gal/ hinfo_dir=gal/ out_dir=ps/ \
          > "ps_${tag}_$nf.log" 2>&1
    done
  done
done

python3 - "$W" <<'EOF'
import sys, glob, h5py, numpy as np
W = sys.argv[1]
ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)

with h5py.File(f"{W}/soap_al1.hdf5") as s, h5py.File(f"{W}/al1_BoundSubhalo/cat/galaxies_78.hdf5") as c:
    n_unmasked = int(np.count_nonzero(s["BoundSubhalo/NumberOfStarParticles"][()] >= 100))
    check(len(c["Mass"]) == n_unmasked, f"masked tensors dropped ({len(c['Mass'])} == {n_unmasked})")
    t = s["BoundSubhalo/StellarInertiaTensorNoniterative"][()][c["SOAPIndex"][()]].astype(float)
    e1 = (t[:, 0] - t[:, 1]) / (t[:, 0] + t[:, 1])
    check(np.allclose(c["e1z"][()], e1), "e1z = (Ixx-Iyy)/(Ixx+Iyy)")
    check(np.isclose(c.attrs["BoxSize"], 300 * 0.681), "box converted to Mpc/h")
    p = s["InputHalos/HaloCentre"][()][c["SOAPIndex"][()]] * 0.681
    check(np.allclose(c["Position"][()], p), "positions converted to Mpc/h")
with h5py.File(f"{W}/al1_ProjectedAperture/cat/galaxies_78.hdf5") as c3, \
     h5py.File(f"{W}/al1_BoundSubhalo/cat/galaxies_78.hdf5") as c6:
    check(np.allclose(c3["e1z"][()], c6["e1z"][()], atol=1e-5) and
          np.allclose(c3["e2z"][()], c6["e2z"][()], atol=1e-5),
          "projz 2D tensor gives the same e as the projected 3D tensor")

def load(tag, name, nf=1):
    return np.loadtxt(f"{W}/{tag}/ps/{name}_sn78_nfold{nf}.dat")

for tag in ("al1_BoundSubhalo", "al0_BoundSubhalo"):
    he, hb = load(tag, "PhE0"), load(tag, "PhB0")
    sel = he[:, 0] < 0.5
    snr_e = np.abs(he[sel, 1] / he[sel, 2]).max()
    snr_b = np.abs(hb[sel, 1] / hb[sel, 2]).max()
    print(f"  {tag}: max |PhE0|/sigma = {snr_e:.1f}, max |PhB0|/sigma = {snr_b:.1f}")
    if tag.startswith("al1"):
        check(snr_e > 10 and snr_b < 5, "aligned sample: strong PhE0, no PhB0")
    else:
        check(snr_e < 5, "random sample: PhE0 consistent with zero")
n1 = len(glob.glob(f"{W}/al1_BoundSubhalo/ps/*.dat"))
n2 = len(glob.glob(f"{W}/al1_BoundSubhalo/ps/2d/*.dat"))
check(n1 == 32 and n2 == 12, f"16 1D + 6 2D outputs per nfold, 2 folds (got {n1} + {n2})")
sys.exit(0 if ok else 1)
EOF
grep -q "ANALYTIC" "$W/ps_al1_BoundSubhalo_1.log" || { echo "FAIL: analytic shape noise not used"; exit 1; }
echo "OK: all stages ran; outputs in $W"
