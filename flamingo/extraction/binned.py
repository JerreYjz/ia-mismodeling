"""1D P(k) and 2D P(k, mu) histogram accumulators (binneddata*.hpp).

Each bin keeps weighted sums  sum(w*x), sum(w*y), sum(w*y^2), sum(w)
(and sum(w*mu) in 2D); ``dump`` writes the means and the error on the mean.
``push_data`` takes whole arrays at once, so the per-mode C loop becomes a
handful of ``np.bincount`` calls.
"""

import numpy as np


def _fmt(v):
    """C++ ostream default formatting (6 significant digits, like %g)."""
    return f"{v:g}"


class BinnedData:
    def __init__(self, nbin, xmin, xmax, logbin=True):
        if logbin and (xmin <= 0 or xmax <= 0):
            raise ValueError("Negative values do not fit in logarithmic binning.")
        self.nbin = int(nbin)
        self.minimum = float(xmin)
        self.maximum = float(xmax)
        self.logbin = bool(logbin)
        n = np.arange(self.nbin)
        if self.logbin:
            self.ratio = (self.maximum / self.minimum) ** (1.0 / self.nbin)
            self.logratio = np.log(self.ratio)
            self.xmin = self.minimum * self.ratio ** n
            self.xmax = self.minimum * self.ratio ** (n + 1)
            self.xcen = self.minimum * self.ratio ** (n + 0.5)
        else:
            self.difference = (self.maximum - self.minimum) / self.nbin
            self.xmin = self.minimum + self.difference * n
            self.xmax = self.minimum + self.difference * (n + 1)
            self.xcen = self.minimum + self.difference * (n + 0.5)
        self.xsum = np.zeros(self.nbin)
        self.ysum = np.zeros(self.nbin)
        self.y2sum = np.zeros(self.nbin)
        self.num = np.zeros(self.nbin)

    def same_binning(self, other):
        return (self.nbin, self.logbin, self.minimum, self.maximum) == \
               (other.nbin, other.logbin, other.minimum, other.maximum)

    def get_bin(self, x):
        """Bin index for each x, -1 outside [min, max)."""
        x = np.asarray(x, dtype=np.float64)
        inside = (x >= self.minimum) & (x < self.maximum)
        b = np.full(x.shape, -1, dtype=np.int64)
        xs = x[inside]
        if self.logbin:
            idx = (np.log(xs / self.minimum) / self.logratio).astype(np.int64)
        else:
            idx = ((xs - self.minimum) / self.difference).astype(np.int64)
        # guard against round-off putting x just below max into bin nbin
        b[inside] = np.minimum(idx, self.nbin - 1)
        return b

    def push_data(self, x, y, w, bins=None):
        """Accumulate arrays x, y with weights w. ``bins`` may be precomputed."""
        if bins is None:
            bins = self.get_bin(x)
        sel = bins >= 0
        b, x, y, w = bins[sel], np.asarray(x)[sel], np.asarray(y)[sel], np.asarray(w)[sel]
        n = self.nbin
        self.xsum += np.bincount(b, w * x, n)
        self.ysum += np.bincount(b, w * y, n)
        self.y2sum += np.bincount(b, w * y * y, n)
        self.num += np.bincount(b, w, n)

    def __iadd__(self, other):
        if not self.same_binning(other):
            raise ValueError("Addition of binned data needs identical binning.")
        self.xsum += other.xsum
        self.ysum += other.ysum
        self.y2sum += other.y2sum
        self.num += other.num
        return self

    def subtract_const(self, p):
        """Subtract a constant (shot-noise floor) from every bin's mean.

        y -> y - P per mode:  ysum -= P*num  and
        y2sum -> sum w (y-P)^2 = y2sum - 2P*ysum + P^2*num.
        (The C version left y2sum unchanged, which corrupted the error
        column of PEE0/PBB0 after the subtraction.)
        """
        filled = self.num > 0
        self.y2sum[filled] += -2.0 * p * self.ysum[filled] + p * p * self.num[filled]
        self.ysum[filled] -= p * self.num[filled]

    def means(self):
        """(k_mean, P_mean, sigma_mean, num) per bin, C formula with 1e-30 guard."""
        den = self.num + 1e-30
        xm = self.xsum / den
        ym = self.ysum / den
        var = np.maximum(self.y2sum / den - ym * ym, 0.0)  # round-off can go < 0
        sig = np.sqrt(var) / np.sqrt(np.maximum(self.num - 1.0, 1e-30))
        return xm, ym, sig, self.num

    def dump(self, fname):
        """Rows ``k  P(k)  sigma_P  N_modes`` for non-empty bins."""
        xm, ym, sig, num = self.means()
        with open(fname, "w") as f:
            for n in range(self.nbin):
                if num[n] == 0:
                    continue
                f.write(f"{_fmt(xm[n])}  {_fmt(ym[n])}  {_fmt(sig[n])}  {_fmt(num[n])}\n")


class BinnedData2D:
    def __init__(self, nbin, xmin, xmax, logbin, munbin):
        self.k = BinnedData(nbin, xmin, xmax, logbin)  # k-binning only
        self.nbin = self.k.nbin
        self.munbin = int(munbin)
        self.mudifference = 1.0 / self.munbin
        m = np.arange(self.munbin)
        self.muxmin = self.mudifference * m
        self.muxmax = self.mudifference * (m + 1)
        self.muxcen = self.mudifference * (m + 0.5)
        shape = (self.nbin, self.munbin)
        self.xsum = np.zeros(shape)
        self.muxsum = np.zeros(shape)
        self.ysum = np.zeros(shape)
        self.y2sum = np.zeros(shape)
        self.num = np.zeros(shape)

    def get_max(self):
        return self.k.maximum

    def get_mubin(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        b = np.full(mu.shape, -1, dtype=np.int64)
        inside = (mu >= 0) & (mu <= 1)
        b[inside] = np.minimum((mu[inside] / self.mudifference).astype(np.int64), self.munbin - 1)
        b[np.abs(mu - 1.0) <= np.finfo(float).eps * np.maximum(1.0, mu)] = self.munbin - 1
        return b

    def get_bin(self, x, mu):
        """Flat (k, mu) bin index, -1 if either is out of range."""
        bk = self.k.get_bin(x)
        bm = self.get_mubin(mu)
        return np.where((bk >= 0) & (bm >= 0), bk * self.munbin + bm, -1)

    def push_data(self, x, mu, y, w, bins=None):
        if bins is None:
            bins = self.get_bin(x, mu)
        sel = bins >= 0
        b = bins[sel]
        x, mu, y, w = (np.asarray(a)[sel] for a in (x, mu, y, w))
        n = self.nbin * self.munbin
        shape = (self.nbin, self.munbin)
        self.xsum += np.bincount(b, w * x, n).reshape(shape)
        self.muxsum += np.bincount(b, w * mu, n).reshape(shape)
        self.ysum += np.bincount(b, w * y, n).reshape(shape)
        self.y2sum += np.bincount(b, w * y * y, n).reshape(shape)
        self.num += np.bincount(b, w, n).reshape(shape)

    def __iadd__(self, other):
        if not (self.k.same_binning(other.k) and self.munbin == other.munbin):
            raise ValueError("Addition of binned data needs identical binning.")
        for name in ("xsum", "muxsum", "ysum", "y2sum", "num"):
            getattr(self, name)[...] += getattr(other, name)
        return self

    def subtract_const(self, p):
        filled = self.num > 0
        self.y2sum[filled] += -2.0 * p * self.ysum[filled] + p * p * self.num[filled]
        self.ysum[filled] -= p * self.num[filled]

    def dump(self, fname):
        """Five nbin x munbin blocks: <k>, <mu>, <P>, sigma_P, N (0 for empty bins)."""
        den = self.num + 1e-30
        empty = self.num == 0
        ym = self.ysum / den
        var = np.maximum(self.y2sum / den - ym * ym, 0.0)
        sig = np.sqrt(var) / np.sqrt(np.maximum(self.num - 1.0, 1e-30))
        blocks = [self.xsum / den, self.muxsum / den, ym, sig]
        with open(fname, "w") as f:
            for blk in blocks:
                blk = np.where(empty, 0.0, blk)
                for row in blk:
                    f.write("".join(_fmt(v) + "  " for v in row) + "\n")
                f.write("\n")
            for row in self.num:
                f.write("".join(_fmt(v) + "  " for v in row) + "\n")
