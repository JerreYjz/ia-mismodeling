#!/usr/bin/env python3
"""k_max where a fitted IA parameter leaves its low-k plateau for good.

Reads fits.csv of fit_stitched_kmax.py (no refitting). For each model,
the plateau at k_i is the mean of the fits at all smaller k_max,
pbar_i = mean(p_0 .. p_{i-1}), and the departure from it, in units of the
parameter's error at k_i,

    dev_i = (p_i - pbar_i) / sigma_p(k_i),

is scanned from low to high k_max. The trend starts at the first k_i with
|dev_i| >= --thr (0.25 sigma) whose next --persist (2) k_max also stay
beyond --thr from the same plateau pbar_i, on the same side: the parameter
has left its plateau and does not come back, so one-point wiggles are
ignored. Using the running plateau, not the first fit alone, catches a
decline that sets in after a small rise. The fits at
neighbouring k_max share most of their data, so the path p(k_max) is much
smoother than sigma_p and a 0.25 sigma departure is a clear trend (the
threshold reproduces cuts drawn by eye at z = 1). The cut is placed
between k_{i-1} and k_i (geometric mean; --cut-at last: at k_{i-1}).

NLA uses a1. TATT uses a1 by default: a2 is nearly unconstrained at low
k_max (it jumps by +-3 sigma between neighbouring k_max while degenerate
with b_TA), so it has no plateau to leave; --tatt-pars a1 a2 adds it (the
earlier onset wins). No onset: stable to the end of the grid (a lower
bound, marked "none").

    python trend_cut.py stitch_kmax/fits.csv --out stitch_kmax/trend

Writes trend_cut.pdf (summary first, then one page per snapshot),
trend_summary.png, trend_cut.csv and kmax_z_trend_<model>.txt.
"""
import argparse
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import quad

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
PARS = {"NLA": ["a1"], "TATT": ["a1", "a2"]}
OM = 0.306


def load(path):
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        rows[(int(r["sn"]), r["model"])].append(
            {k: (float(v) if k not in ("model",) else v) for k, v in r.items()})
    for v in rows.values():
        v.sort(key=lambda r: r["kmax"])
    return rows


def onset(k, p, e, thr, persist, frac=None):
    """Index of the first k_max that leaves the running plateau for good.
    frac: threshold as a fraction of the plateau value instead of sigma."""
    for i in range(1, len(p)):
        pbar = np.mean(p[:i])
        scale = abs(pbar)*frac/thr if frac else e[i:i + persist + 1]
        dev = (p[i:i + persist + 1] - pbar)/scale
        if abs(dev[0]) >= thr and np.all(dev*np.sign(dev[0]) >= thr):
            return i
    return None


def cut_k(k, i, mode):
    if i is None:
        return float(k[-1]), "none"
    return (float(np.sqrt(k[i - 1]*k[i])) if mode == "mid" else float(k[i - 1])), "cut"


def growth(z):
    E = lambda a: np.sqrt(OM/a**3 + 1 - OM)
    g = lambda a: 2.5*OM*E(a)*quad(lambda x: 1/(x*E(x))**3, 1e-6, a)[0]
    return np.array([g(1/(1 + x))/g(1) for x in np.atleast_1d(z)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fits")
    ap.add_argument("--out", default="trend")
    ap.add_argument("--thr", type=float, default=0.25,
                    help="departure from the k_ref value, in sigma")
    ap.add_argument("--thr-frac", type=float, default=None,
                    help="threshold as a fractional change of a1 (e.g. 0.01) "
                         "instead of --thr sigma; for a1 only")
    ap.add_argument("--persist", type=int, default=2,
                    help="following k_max that must stay beyond --thr")
    ap.add_argument("--tatt-pars", nargs="+", default=["a1"], choices=["a1", "a2"])
    ap.add_argument("--cut-at", choices=["mid", "last"], default="mid")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    PARS["TATT"] = a.tatt_pars
    rows = load(a.fits)
    sns = sorted({s for s, _ in rows}, key=lambda s: rows[(s, "NLA")][0]["z"])
    table = []
    for sn in sns:
        t = {"sn": sn, "z": rows[(sn, "NLA")][0]["z"], "cut": {}, "by": {}}
        for m in ("NLA", "TATT"):
            rs = rows[(sn, m)]
            k = np.array([r["kmax"] for r in rs])
            best = (np.inf, "none", "-")
            for par in ["a1", "a2"] if m == "TATT" else ["a1"]:
                p = np.array([r[par] for r in rs])
                e = np.array([r[par + "_err"] for r in rs])
                fr = a.thr_frac if par == "a1" else None
                kc, kind = cut_k(k, onset(k, p, e, a.thr, a.persist, fr), a.cut_at)
                t.setdefault("each", {})[(m, par)] = (kc, kind)
                if par in PARS[m] and kind == "cut" and kc < best[0]:
                    best = (kc, kind, par)
            if best[1] == "none":
                best = (float(k[-1]), "none", "-")
            t["cut"][m] = best[:2]
            t["by"][m] = best[2]
        table.append(t)

    print(f"leaves the running plateau by >= "
          f"{(str(100*a.thr_frac) + '% (a1)') if a.thr_frac else str(a.thr) + ' sigma'} for the next {a.persist} "
          f"k_max; TATT cut from {a.tatt_pars} ('+': stable to the end of the grid)")
    print("   z     NLA (a1)     TATT a1 / (a2, not used unless asked) -> TATT cut")
    for t in table:
        e = t["each"]
        f = lambda kc, kind: f"{kc:.2f}{'+' if kind == 'none' else ' '}"
        print(f"{t['z']:5.2f}   {f(*t['cut']['NLA'])}        "
              f"{f(*e[('TATT', 'a1')])} / {f(*e[('TATT', 'a2')])} -> {f(*t['cut']['TATT'])}")

    z = np.array([t["z"] for t in table])
    o = np.argsort(z)
    with open(os.path.join(a.out, "trend_cut.csv"), "w") as fh:
        fh.write("sn,z,k_NLA,kind_NLA,k_TATT,kind_TATT,TATT_set_by,k_TATT_a1,k_TATT_a2\n")
        for t in table:
            fh.write(f"{t['sn']},{t['z']},{t['cut']['NLA'][0]:.4f},{t['cut']['NLA'][1]},"
                     f"{t['cut']['TATT'][0]:.4f},{t['cut']['TATT'][1]},{t['by']['TATT']},"
                     f"{t['each'][('TATT', 'a1')][0]:.4f},{t['each'][('TATT', 'a2')][0]:.4f}\n")
    for m in ("NLA", "TATT"):
        np.savetxt(os.path.join(a.out, f"kmax_z_trend_{m.lower()}.txt"),
                   [[z[i], table[i]["cut"][m][0]] for i in o], fmt="%.3f %.4f",
                   header=f"z  k_max [h/Mpc]: onset of a steady trend, {m} "
                          "(values at the grid end are lower bounds)")

    pdf = PdfPages(os.path.join(a.out, "trend_cut.pdf"))
    # --- summary
    zz = np.linspace(0, z.max(), 120)
    Dz = growth(zz)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m in ("NLA", "TATT"):
        kc = np.array([t["cut"][m][0] for t in table])
        nn = np.array([t["cut"][m][1] == "none" for t in table])
        ax.plot(z[o], kc[o], "-o", color=COL[m], lw=2, label=f"{m}: trend onset")
        if nn.any():
            ax.plot(z[nn], kc[nn], "^", ms=9, mfc="none", color="k")
    try:
        import fit_stitched_kmax as S
        ax.plot(z[o], [S.KT[table[i]["sn"]] for i in o], "-", color="#e76f51", lw=1,
                label="stitch k_t")
    except Exception:
        pass
    ax.plot(zz, 0.35/Dz, ":", color="0.4", label="0.35/D(z)")
    ax.set_yscale("log")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    thrtxt = f"{100*a.thr_frac:g}%" if a.thr_frac else f"{a.thr} sigma"
    ax.set_title(f"k_max where a1 leaves its low-k plateau by {thrtxt} for good "
                 f"(TATT from {'+'.join(a.tatt_pars)})\n"
                 r"$\triangle$: stable to the end of the grid", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "trend_summary.png"), dpi=130)
    pdf.savefig(fig)
    plt.close(fig)
    # --- per snapshot
    for t in table:
        sn = t["sn"]
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
        for ax, (m, par) in zip(axs, [("NLA", "a1"), ("TATT", "a1"), ("TATT", "a2")]):
            rs = rows[(sn, m)]
            k = np.array([r["kmax"] for r in rs])
            p = np.array([r[par] for r in rs])
            e = np.array([r[par + "_err"] for r in rs])
            ax.errorbar(k, p, e, fmt="o", ms=4, color=COL[m], alpha=0.85)
            ax.plot(k, p, "-", color=COL[m], lw=0.8, alpha=0.5)
            kc, kind = t["each"][(m, par)]
            if kind == "cut":
                ax.axvline(kc, color="#f4a261" if m == "NLA" else "#2a9d8f", lw=2,
                           label=f"trend onset: {kc:.2f}")
            if m == "TATT" and t["cut"]["TATT"][1] == "cut":
                ax.axvline(t["cut"]["TATT"][0], color="k", ls=":", lw=1,
                           label=f"TATT cut ({t['by']['TATT']}): {t['cut']['TATT'][0]:.2f}")
            ax.set_xscale("log")
            ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
            ax.set_title(f"{m}: {par}" + ("" if kind == "cut" else "  (no steady trend)"))
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            # zoom on the low-k values, where the onset is decided
            lo = p[k <= max(kc, 0.3)*1.6] if kind == "cut" else p
            pad = max(np.ptp(lo), np.median(e)*0.5)
            ax.set_ylim(np.min(lo) - 1.5*pad, np.max(lo) + 1.5*pad)
        fig.suptitle(f"sn{sn}  z = {t['z']:.2f}:  NLA cut {t['cut']['NLA'][0]:.2f}, "
                     f"TATT cut {t['cut']['TATT'][0]:.2f} h/Mpc "
                     f"(y zoomed on the range around the onset)")
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(os.path.join(a.out, f"trend_sn{sn}.png"), dpi=100)
        plt.close(fig)
    pdf.close()
    print(f"wrote {a.out}/trend_cut.pdf (summary first), trend_summary.png, "
          "trend_cut.csv, kmax_z_trend_<model>.txt")


if __name__ == "__main__":
    main()
