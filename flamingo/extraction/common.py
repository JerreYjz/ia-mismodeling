"""Shared helpers for the ia-power-shi Python pipeline.

Contents
--------
* ``read_parameter``   : Gadget-style "Tag  value" parameter-file reader
                         (replaces read_parameter() in the C io_input.c files).
* ``c_atof/c_atoi``    : C-compatible number parsing ("1e4." -> 10000.0).
* FLAMINGO readers     : local/remote HDF5 access, header, chunked row reads.
* ``new_grid`` / ``fft_r2c_inplace`` / ``save_grid``
                       : the FieldData<float> pieces used upstream of Pds.
                         The grid uses FFTW's in-place padded layout
                         (ng, ng, 2*(ng//2+1)) float32, so after the FFT the
                         same buffer *is* the (ng, ng, ng//2+1) complex64
                         array that gets written to disk, byte-for-byte the
                         format the C code wrote and Pds reads.
"""

import os
import re
import sys
import time

import numpy as np

try:
    import scipy.fft as _fft
except ImportError:  # pragma: no cover
    _fft = None


# ---------------------------------------------------------------------------
# number parsing with C atof/atoi semantics
# ---------------------------------------------------------------------------
_FLOAT_RE = re.compile(r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_INT_RE = re.compile(r"^\s*[+-]?\d+")


def c_atof(s):
    """Parse the longest numeric prefix like C's atof (``'1e4.'`` -> 1e4).

    The existing parameter files contain values such as ``1e-1.`` which
    Python's ``float()`` rejects but C silently accepts.
    """
    m = _FLOAT_RE.match(s)
    return float(m.group(0)) if m else 0.0


def c_atoi(s):
    m = _INT_RE.match(s)
    return int(m.group(0)) if m else 0


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Gadget-style parameter file:   Tag   value      ('%' or '#' starts a comment)
# ---------------------------------------------------------------------------
def read_parameter(fname, tags, optional=None):
    """Read a "Tag value" parameter file.

    ``tags`` maps tag name -> type (``str``, ``int`` or ``float``); each must
    appear exactly once. ``optional`` maps tag name -> (type, default).
    Unknown or repeated tags are an error. Returns a dict tag -> value.
    """
    optional = optional or {}
    params = {}
    errors = []
    if not os.path.isfile(fname):
        sys.exit(f"Parameter file {fname} not found.")

    with open(fname) as fd:
        for line in fd:
            tok = line.split()
            if len(tok) < 2 or tok[0][0] in "%#":
                continue
            key, val = tok[0], tok[1]
            if (key not in tags and key not in optional) or key in params:
                errors.append(f"Error in file {fname}:   Tag '{key}' not allowed or multiple defined.")
                continue
            kind = tags[key] if key in tags else optional[key][0]
            if kind is int:
                params[key] = c_atoi(val)
            elif kind is float:
                params[key] = c_atof(val)
            else:
                params[key] = val
            print(f"{key:>35s}\t{val:>10s}")

    for key in tags:
        if key not in params:
            errors.append(f"Error. I miss a value for tag '{key}' in parameter file '{fname}'.")
    for key, (_, default) in optional.items():
        params.setdefault(key, default)
    if errors:
        # The C version called exit(0) here, i.e. reported *success* on a bad
        # parameter file; exit non-zero instead.
        sys.exit("\n".join(errors))
    return params


# ---------------------------------------------------------------------------
# FLAMINGO file access (local h5py file or remote hdfstream file)
# ---------------------------------------------------------------------------
REMOTE_PREFIX = "cosma:"
ROWS_PER_READ = 1 << 24          # rows per chunked read of a catalogue/snapshot dataset


def open_hdf5(path):
    """Open a FLAMINGO HDF5 file.

    ``path`` is either a local file (read with h5py) or
    ``cosma:/FLAMINGO/<box>/<run>/...`` to stream it from the COSMA data
    server with the ``hdfstream`` module (pip install hdfstream). Both return
    an object with the h5py interface used here (``f[name][a:b]``, ``.attrs``,
    ``.shape``).
    """
    if path.startswith(REMOTE_PREFIX):
        import hdfstream

        root = hdfstream.open("cosma", "/")
        return root[path[len(REMOTE_PREFIX):].lstrip("/")]
    import h5py

    if not os.path.isfile(path):
        sys.exit(f"HDF5 file {path} not found.")
    return h5py.File(path, "r")


def attr_scalar(attrs, key):
    """HDF5 attribute as a Python scalar (FLAMINGO stores most as 1-element arrays)."""
    v = np.asarray(attrs[key])
    v = v.reshape(-1)[0] if v.ndim else v[()]
    return v.decode() if isinstance(v, bytes) else v


def flamingo_header(f):
    """(BoxSize [comoving Mpc], h, redshift) from a SOAP file or a snapshot.

    FLAMINGO lengths are comoving Mpc and masses 1e10 Msun, both *without*
    h (unlike TNG's ckpc/h and 1e10 Msun/h).
    """
    box = float(np.asarray(f["Header"].attrs["BoxSize"]).reshape(-1)[0])
    h = float(attr_scalar(f["Cosmology"].attrs, "h"))
    z = float(attr_scalar(f["Header"].attrs, "Redshift"))
    return box, h, z


def read_rows(dset, start, stop, idx=None):
    """Rows ``start:stop`` of a dataset, optionally only local indices ``idx``.

    For a remote dataset and a sparse selection only the selected runs of
    rows are requested (one HTTP request via ``request_slices``), which is
    much less data than the whole block; locally the block is read and
    indexed.
    """
    if idx is None:
        return np.asarray(dset[start:stop])
    if len(idx) == 0:
        return np.asarray(dset[start:start])
    if hasattr(dset, "request_slices") and len(idx) < 0.2 * (stop - start):
        # contiguous runs of selected rows
        brk = np.flatnonzero(np.diff(idx) != 1) + 1
        lo = idx[np.concatenate([[0], brk])]
        hi = idx[np.concatenate([brk - 1, [len(idx) - 1]])] + 1
        if len(lo) <= 20000:
            nd = len(dset.shape)
            slices = [(np.s_[start + a:start + b],) + (np.s_[:],) * (nd - 1) for a, b in zip(lo, hi)]
            return np.asarray(dset.request_slices(slices))
    return np.asarray(dset[start:stop])[idx]


# ---------------------------------------------------------------------------
# Grid in FFTW in-place layout (FieldData<float> equivalent)
# ---------------------------------------------------------------------------
def n_threads():
    """FFT threads: OMP_NUM_THREADS like the C code, else all cores."""
    env = os.environ.get("OMP_NUM_THREADS")
    return c_atoi(env) if env else (os.cpu_count() or 1)


def new_grid(ng):
    """Zeroed real-space grid, padded to (ng, ng, 2*(ng//2+1)) float32.

    Use ``grid[..., :ng]`` for the real-space values.
    """
    return np.zeros((ng, ng, 2 * (ng // 2 + 1)), dtype=np.float32)


def fft_r2c_inplace(grid):
    """Forward real-to-complex 3D FFT in place, normalised by 1/ng^3.

    Same convention as the C ``FieldData::do_fft`` (FFTW forward sign,
    exp(-i k.x), divided by ng^3). Peak memory stays ~1 grid plus one slab.
    Returns the complex64 view of shape (ng, ng, ng//2+1).
    """
    ng = grid.shape[0]
    workers = n_threads()
    log(f"FFT by {workers} threads ...")
    cgrid = grid.view(np.complex64)
    # 2D r2c on each x-slab: slab i of the complex view occupies exactly the
    # bytes of slab i of the real grid, which is already consumed.
    for i in range(ng):
        cgrid[i] = _fft.rfft2(grid[i, :, :ng], workers=workers)
    # 1D c2c along x, in column blocks to keep the temporary small.
    step = max(1, (64 << 20) // (ng * cgrid.shape[2] * 8))
    for j0 in range(0, ng, step):
        blk = cgrid[:, j0:j0 + step, :]
        blk[...] = _fft.fft(blk, axis=0, workers=workers, overwrite_x=True)
    cgrid *= np.float32(1.0 / float(ng) ** 3)
    log("FFT and normalization done.")
    return cgrid


def save_grid(cgrid, fname):
    """Write the complex grid as raw complex64 (the C .bin format)."""
    d = os.path.dirname(fname)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = fname + ".part"
    with open(tmp, "wb") as f:
        cgrid.tofile(f)
    os.replace(tmp, fname)  # never leave a truncated file under the final name
    log(f"wrote {cgrid.nbytes} bytes to {fname}")


def periodic_index(u, ng):
    """floor(u) wrapped into [0, ng) as int64 (handles folding and negatives)."""
    return np.floor(u).astype(np.int64) % ng


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        print(f"Seconds used now: {time.time() - self.t0:.2f}; {msg}", flush=True)
