#!/usr/bin/env python3
"""data.json of plots/cp_fisher_k0scan from fisher_minimize.json.

Step 2 of RERUN_PLAN_NBall.md: the Fisher-only figure of bias (Kable, Addison
& Bennett 2019, arXiv:1908.01626; flat priors as Gaussians, CMB prior) on the
CP truth, against the xi_+- cut K0, for the two IA sets: set 1 NLA (2) and
TATT (5 parameters, power law in z), set 2 NLAz (A1 per source bin) and TATTz
(A1, A2 per bin, one b_TA). Four setups: cosmic shear, 3x2pt with gamma_t
cut at 0.3 and at 0.5 (w at 0.2), 2x2pt (gamma_t and w cut at K0). Next to
each CP point, the stored fob_mean_linear of the S1 run with the same key
(same estimator, the old truth with the close-pair term subtracted), where
it exists. Per point also sigma(S8), the S8 shift, the reference-fit chi2,
the IA parameters at a bound of the reference fit and those the data do not
constrain. The S1 cs NLAz runs (K0 0.2, 0.25) are left out: their reference
fit sat in the A1_8 ~ 0 basin (fisher_minimize.trend_restart).

    python make_cp_fisher_figure.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "cp_fisher_k0scan")
K0 = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
MODELS = ["NLA", "TATT", "NLAz", "TATTz"]
SETUPS = {"cs": ("cs+cmb", ""), "3x2 gt0.3": ("3x2+cmb", ""), "3x2 gt0.5": ("3x2+cmb", "|gt0.5"),
          "2x2": ("2x2+cmb", "")}


def main():
    R = json.load(open(os.path.join(HERE, "fisher_minimize.json")))["runs"]
    lines = {}
    for su, (pre, suf) in SETUPS.items():
        for m in MODELS:
            row = {q: [] for q in ("fob", "fob_S1", "sigma_S8", "dS8", "chi2_ref", "npts",
                                   "at_bound", "no_info")}
            for K in K0:
                r = R.get(f"{pre}|{m}|{K:g}|CP{suf}|fisher")
                s = None if (su == "cs" and m in ("NLAz", "TATTz")) else R.get(f"{pre}|{m}|{K:g}|S1{suf}")
                row["fob_S1"].append(None if s is None else s["flatprior"]["fob_mean_linear"])
                if r is None:
                    for q in ("fob", "sigma_S8", "dS8", "chi2_ref", "npts", "at_bound", "no_info"):
                        row[q].append(None)
                    continue
                row["fob"].append(r["fob"])
                row["sigma_S8"].append(r["flatprior"]["sigma"][1])
                row["dS8"].append(r["flatprior"]["shift_mean_linear"][1])
                row["chi2_ref"].append(r["chi2_ref"])
                row["npts"].append(r["npts"])
                row["at_bound"].append([p.replace("roman_", "") for p in r["at_bound_ref"]])
                row["no_info"].append([p.replace("roman_", "") for p in r.get("no_info", [])])
            lines[f"{su}|{m}"] = row
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(about=__doc__, K0=K0, threshold=0.3, setups=list(SETUPS), models=MODELS, lines=lines),
              open(os.path.join(OUT, "data.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "data.json"))


if __name__ == "__main__":
    main()
