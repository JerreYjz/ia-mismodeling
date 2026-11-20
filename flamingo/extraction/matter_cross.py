#!/usr/bin/env python3
"""Galaxy-matter cross spectrum P_gm for FLAMINGO L1_m8, from the full dark matter.

No file of the FLAMINGO data server holds a galaxy-matter spectrum, and the
reduced snapshots keep particles by halo membership (not a random subsample),
so the matter field is built from every dark-matter particle of a full
snapshot, streamed from the COSMA server (about 7.5 h per snapshot at
1.7e6 rows/s). Two stages:

  stream   reads PartType1/Coordinates in blocks and deposits each particle
           (NGP) on three count grids, the box folded nfold = 1, 4 and 16
           times (ng = 1024, 1024, 512), the folding of galaxy_ds_field.py.
           A checkpoint (grids plus the next row) is written every 30 min, so
           an interrupted run resumes where it stopped; a failed read is
           retried with backoff until the server answers again.
  power    for each fold, the dark-matter contrast delta_m = n/<n> - 1 and the
           galaxy contrast delta_g of galaxy_ds_field.py (the same galaxies
           and selection as Phh), FFT with common.fft_r2c_inplace, and the
           three spectra P_mm, P_gm, P_gg binned exactly as ia_power.py
           (30 log bins in 0.003 to 20 h/Mpc, k_z = 0 modes weighted 1/2, the
           Nyquist plane dropped). No shot-noise subtraction and no window
           deconvolution, as for Phh; the Poisson levels V/N are printed and
           written to the info file.

Outputs (k in h/Mpc, P in (Mpc/h)^3, rows "k P sigma_P N_modes" as Phh):
    <out>/Pmm_dm_sn<sn>_nfold<f>.dat, Pgm_sn<sn>_nfold<f>.dat, Pgg_sn<sn>_nfold<f>.dat
    <out>/matter_cross_sn<sn>.json   (N_dm, N_gal, V/N, redshift, folds)

The cross is with the dark matter only (PartType1), not the total matter;
baryons change the matter spectrum by a few percent below k ~ 1 h/Mpc, and
ratios such as b_x = P_gm/P_mm and r = P_gm/sqrt(P_gg P_mm) use the same
dark-matter field throughout.

    python matter_cross.py stream <sn> [--max-rows N]
    python matter_cross.py power  <sn>
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np

from binned import BinnedData
from common import fft_r2c_inplace, log, new_grid, periodic_index

SNAP = "cosma:/FLAMINGO/L1_m8/L1_m8/snapshots/flamingo_{sn:04d}/flamingo_{sn:04d}.hdf5"
RUN = os.path.expanduser("~/ia_power_flamingo/run_L1_m8_particles/results")
GRID_DIR = os.path.join(RUN, "DM_Field")
GAL_DIR = os.path.join(RUN, "Galaxies")
OUT_DIR = os.path.join(RUN, "PS_matter")
FOLDS = [(1, 1024), (4, 1024), (16, 512)]
BLOCK = 1 << 23                 # rows per streamed block (200 MB of float64 coordinates)
CKPT_SEC = 1800.0
NREAD = 3                       # parallel block requests
KMIN, KMAX, NBIN = 0.003, 20.0, 30


def grid_name(sn, f, ng, tag=""):
    return os.path.join(GRID_DIR, f"dm_counts{tag}_sn{sn}_nfold{f}_ng{ng}.npy")


def save_atomic(arr, fname):
    tmp = fname + ".part.npy"
    np.save(tmp, arr)
    os.replace(tmp, fname)


# ---------------------------------------------------------------------------
# stage 1: stream the dark matter onto count grids
# ---------------------------------------------------------------------------
def read_block(dset, a, b):
    """Rows a:b, retried with backoff (at most 10 min apart) for as long as it
    takes: a server outage (the COSMA server shut down for about two hours on
    2026-11-19) stalls the stream instead of ending it, and the rows already
    deposited stay on the grids."""
    attempt = 0
    while True:
        try:
            return np.asarray(dset[a:b])
        except Exception as e:
            wait = min(600, 10 * 2 ** attempt)
            attempt += 1
            log(f"read {a}:{b} failed ({type(e).__name__}: {str(e)[:120]}); retry {attempt} in {wait} s")
            time.sleep(wait)


def reader(dset, start, stop, q, stop_flag):
    """Producer thread: blocks in order, NREAD requests in flight at once
    (one request reaches about 1.3e6 rows/s, three 2.0e6; six gave no more)."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    starts = iter(range(start, stop, BLOCK))
    pending = deque()
    try:
        with ThreadPoolExecutor(NREAD) as ex:
            for a in starts:
                pending.append((a, ex.submit(read_block, dset, a, min(a + BLOCK, stop))))
                if len(pending) >= NREAD:
                    break
            while pending and not stop_flag.is_set():
                a, fut = pending.popleft()
                q.put((a, min(a + BLOCK, stop), fut.result()))
                nxt = next(starts, None)
                if nxt is not None:
                    pending.append((nxt, ex.submit(read_block, dset, nxt, min(nxt + BLOCK, stop))))
    except Exception as e:
        q.put(e)
        return
    q.put(None)


def stream(sn, max_rows=None):
    from common import flamingo_header, open_hdf5

    os.makedirs(GRID_DIR, exist_ok=True)
    f = open_hdf5(SNAP.format(sn=sn))
    box, h, z = flamingo_header(f)
    dset = f["PartType1/Coordinates"]
    ntot = int(dset.shape[0])
    stop = ntot if max_rows is None else min(ntot, max_rows)
    tag = "" if max_rows is None else "_test"
    box_h = box * h
    log(f"snapshot {sn}: z = {z:.3f}, box = {box} Mpc ({box_h:.4f} Mpc/h), N_dm = {ntot:,}; "
        f"streaming rows 0:{stop:,}")

    ck = os.path.join(GRID_DIR, f"dm_ckpt{tag}_sn{sn}.json")
    if os.path.exists(ck):
        state = json.load(open(ck))
        grids = [np.load(grid_name(sn, fo, ng, tag + "_ckpt")) for fo, ng in FOLDS]
        start = state["next_row"]
        log(f"resuming at row {start:,} from {ck}")
    else:
        grids = [np.zeros(ng ** 3, dtype=np.float32) for _, ng in FOLDS]
        start = 0

    q = queue.Queue(maxsize=1)
    stop_flag = threading.Event()
    th = threading.Thread(target=reader, args=(dset, start, stop, q, stop_flag), daemon=True)
    th.start()
    t0 = t_ck = time.time()
    done0 = start
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        a, b, x = item
        # positions in Mpc -> Mpc/h, then cells of the folded box (galaxy_ds_field.py)
        pos = x * h
        for g, (fo, ng) in zip(grids, FOLDS):
            c = periodic_index(pos * (ng / (box_h / fo)) + 1e-10, ng)
            np.add.at(g, (c[:, 0] * ng + c[:, 1]) * ng + c[:, 2], np.float32(1.0))
        rate = (b - done0) / (time.time() - t0)
        if (b // BLOCK) % 25 == 0 or b == stop:
            log(f"  rows {b:,}/{stop:,} ({100 * b / stop:.1f}%), {rate:.2e} rows/s, "
                f"{(stop - b) / rate / 3600:.2f} h left")
        if time.time() - t_ck > CKPT_SEC and b < stop:
            for g, (fo, ng) in zip(grids, FOLDS):
                save_atomic(g, grid_name(sn, fo, ng, tag + "_ckpt"))
            json.dump(dict(next_row=b, sn=sn, z=z, N_dm=ntot), open(ck + ".part", "w"))
            os.replace(ck + ".part", ck)
            t_ck = time.time()
            log(f"  checkpoint at row {b:,}")

    for g, (fo, ng) in zip(grids, FOLDS):
        s = float(g.sum(dtype=np.float64))
        assert abs(s - stop) < 0.5, f"fold {fo}: {s} particles deposited, expected {stop}"
        save_atomic(g, grid_name(sn, fo, ng, tag))
    json.dump(dict(sn=sn, z=z, box_Mpc=box, h=h, N_dm=stop, N_dm_snapshot=ntot,
                   folds=FOLDS, seconds=time.time() - t0),
              open(os.path.join(GRID_DIR, f"dm_info{tag}_sn{sn}.json"), "w"), indent=1)
    for fo, ng in FOLDS:
        p = grid_name(sn, fo, ng, tag + "_ckpt")
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(ck):
        os.remove(ck)
    log(f"snapshot {sn}: {stop:,} particles on the grids in {(time.time() - t0) / 3600:.2f} h")


# ---------------------------------------------------------------------------
# stage 2: P_mm, P_gm, P_gg
# ---------------------------------------------------------------------------
def galaxy_cells(sn, ng, nfold):
    """Cells of the Phh galaxies (galaxy_ds_field.py: finite shapes, Select all)."""
    import h5py

    with h5py.File(os.path.join(GAL_DIR, f"galaxies_{sn}.hdf5"), "r") as fh:
        pos = fh["Position"][()].astype(np.float64)
        e1, e2 = fh["e1z"][()], fh["e2z"][()]
        box_h, z = float(fh.attrs["BoxSize"]), float(fh.attrs["Redshift"])
    used = np.isfinite(e1) & np.isfinite(e2)
    c = periodic_index(pos[used] * (ng / (box_h / nfold)) + 1e-10, ng)
    return c, box_h, z


def contrast_grid(ng, deposit):
    """Padded grid filled by deposit(grid) -> N, delta = n/<n> - 1, then FFT."""
    grid = new_grid(ng)
    n = deposit(grid)
    real = grid[..., :ng]
    real *= np.float32(float(ng) ** 3 / n)
    real -= np.float32(1.0)
    return fft_r2c_inplace(grid), n


def signed(n, ng):
    return np.where(n > ng // 2, n - ng, n).astype(np.float64)


def spectra(cm, cg, boxlen, nfold):
    """P_mm, P_gm, P_gg over the half-complex grid (calc_power of ia_power.py)."""
    ng = cm.shape[0]
    kf = 2 * np.pi / boxlen
    vol = (boxlen * nfold) ** 3
    out = {n: BinnedData(NBIN, KMIN, KMAX, True) for n in ("Pmm", "Pgm", "Pgg")}
    jidx = np.arange(ng)
    jidx = jidx[np.abs(signed(jidx, ng)) * kf <= KMAX]
    kidx = np.arange(ng // 2)
    kidx = kidx[kidx * kf <= KMAX]
    jj = signed(jidx, ng)[:, None]
    kk = kidx.astype(np.float64)[None, :]
    weight = np.broadcast_to(np.where(kk == 0, 0.5, 1.0), (len(jidx), len(kidx))).ravel()
    for i in range(ng):
        ii = float(signed(i, ng))
        if abs(ii) * kf > KMAX:
            continue
        k2 = (ii * ii + jj * jj + kk * kk).ravel()
        valid = k2 > 0
        k = kf * np.sqrt(k2[valid])
        w = weight[valid]
        m = np.asarray(cm[i][np.ix_(jidx, kidx)], dtype=np.complex128).ravel()[valid]
        g = np.asarray(cg[i][np.ix_(jidx, kidx)], dtype=np.complex128).ravel()[valid]
        bins = out["Pmm"].get_bin(k)
        for name, (a, b) in dict(Pmm=(m, m), Pgm=(g, m), Pgg=(g, g)).items():
            out[name].push_data(k, vol * (a.real * b.real + a.imag * b.imag), w, bins=bins)
    return out


def power(sn, tag=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    info = json.load(open(os.path.join(GRID_DIR, f"dm_info{tag}_sn{sn}.json")))
    meta = dict(sn=sn, z=info["z"], N_dm=info["N_dm"], folds={})
    for fo, ng in FOLDS:
        counts = np.load(grid_name(sn, fo, ng, tag), mmap_mode="r").reshape(ng, ng, ng)

        def dep_m(grid):
            for i in range(ng):
                grid[i, :, :ng] = counts[i]
            return float(info["N_dm"])

        cm, n_dm = contrast_grid(ng, dep_m)
        cells, box_h, z = galaxy_cells(sn, ng, fo)
        rowlen = 2 * (ng // 2 + 1)

        def dep_g(grid):
            np.add.at(grid.reshape(-1), (cells[:, 0] * ng + cells[:, 1]) * rowlen + cells[:, 2],
                      np.float32(1.0))
            return float(len(cells))

        cg, n_gal = contrast_grid(ng, dep_g)
        boxlen = box_h / fo
        log(f"sn {sn} nfold {fo}: spectra ...")
        res = spectra(cm, cg, boxlen, fo)
        for name, bd in res.items():
            stem = "Pmm_dm" if name == "Pmm" else name
            bd.dump(os.path.join(OUT_DIR, f"{stem}{tag}_sn{sn}_nfold{fo}.dat"))
        vol = box_h ** 3
        meta["folds"][str(fo)] = dict(ng=ng, k_nyquist=np.pi * ng / boxlen,
                                      shot_dm=vol / n_dm, shot_gal=vol / n_gal, N_gal=int(n_gal))
        meta.update(box_h=box_h, z_gal=z)
        del cm, cg
    json.dump(meta, open(os.path.join(OUT_DIR, f"matter_cross{tag}_sn{sn}.json"), "w"), indent=1)
    log(f"sn {sn}: spectra written to {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stream", "power"])
    ap.add_argument("sn", type=int)
    ap.add_argument("--max-rows", type=int, default=None, help="test: stream only the first N rows")
    a = ap.parse_args()
    if a.stage == "stream":
        stream(a.sn, a.max_rows)
    else:
        power(a.sn, "" if a.max_rows is None else "_test")


if __name__ == "__main__":
    main()
