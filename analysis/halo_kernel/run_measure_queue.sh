#!/bin/bash
# Measure the kernels of every snapshot as soon as its SOAP halo file (with so_tensor) exists.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
PY=~/cocoa/Cocoa/.local/bin/python
H=~/ia_power_flamingo/run_L1_m8_particles/results/SOAP_halos
for sn in "$@"; do
  until $PY -c "import h5py,sys; f=h5py.File('$H/halos_$sn.hdf5','r'); sys.exit(0 if 'so_tensor' in f['halos'] else 1)" 2>/dev/null; do sleep 30; done
  if [ -f kernels/kernels_$sn.npz ] && [ kernels/kernels_$sn.npz -nt $H/halos_$sn.hdf5 ]; then echo "sn $sn done already"; continue; fi
  $PY hk_measure.py $sn 2>&1 | grep -v "FFT\|^\[" > logs_measure_$sn.txt
  echo "sn $sn: $(tail -1 logs_measure_$sn.txt)"
done
echo "QUEUE DONE"
