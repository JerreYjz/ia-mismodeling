"""Turn an MCMC yaml + its finished chain into a Cobaya minimize yaml.

The minimizer starts from the chain's best sample (lowest -log posterior),
with small random offsets for the best_of restarts, and uses the chain's
covariance to scale its steps. The result is the MAP: the projection-free
best fit, the same quantity gn_bias.py measures.

  python make_minimize_yaml.py \\
      --yaml ./projects/roman_real/MCMC_NLA_cs_final_d3a_emul.yaml \\
      --chain projects/roman_real/chains/MCMC_NLA_cs_final_d3a_emul

Writes, next to the input yaml:
  MIN_<name>.yaml          minimize sampler, output chains/MIN_<name>/MIN_<name>
  MIN_<name>.covmat        the chain covariance of the sampled parameters
Run it like the MCMC (mcmc.sbatch), with as many MPI ranks as best_of.
"""
import argparse
import os

import numpy as np
import yaml
from getdist import loadMCSamples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="the MCMC yaml of the chain")
    ap.add_argument("--chain", required=True, help="chain root (output: without ./)")
    ap.add_argument("--burn-in", type=float, default=0.3)
    ap.add_argument("--best-of", type=int, default=4)
    ap.add_argument("--start-scale", type=float, default=0.1,
                    help="restart offsets, in units of the chain sigma")
    ap.add_argument("--rhoend", type=float, default=0.05,
                    help="BOBYQA final trust radius in chain-sigma units "
                         "(Cobaya default 0.05; 0.01 for flat directions)")
    ap.add_argument("--ignore-prior", action="store_true",
                    help="maximum likelihood instead of MAP (default: MAP, "
                         "as gn_bias.py, which keeps the Gaussian priors)")
    a = ap.parse_args()

    info = yaml.safe_load(open(a.yaml))
    s = loadMCSamples(a.chain, settings={"ignore_rows": a.burn_in})
    best = int(np.argmin(s.loglikes))           # getdist loglikes = -log post
    names = {p.name for p in s.getParamNames().names}

    sampled = [k for k, v in info["params"].items()
               if isinstance(v, dict) and "prior" in v]
    miss = [p for p in sampled if p not in names]
    if miss:
        raise SystemExit(f"chain lacks sampled parameters {miss}")
    print(f"best chain sample: -log post = {s.loglikes[best]:.3f}")
    for p in sampled:
        v, sd = float(s[p][best]), float(s.std(p))
        lo = info["params"][p]["prior"].get("min")
        hi = info["params"][p]["prior"].get("max")
        if lo is not None:
            v = max(v, lo + 1e-6*(hi - lo))
        if hi is not None:
            v = min(v, hi - 1e-6*(hi - lo))
        info["params"][p]["ref"] = {"dist": "norm", "loc": v,
                                    "scale": a.start_scale*sd}
        info["params"][p]["proposal"] = sd

    # name from the chain's output name (not the yaml file name), so the
    # minimum lands next to the chain whatever the yaml is called:
    # chains/MCMC_NLA_cs_final_d3a_emul[/...] -> chains/MIN_NLA_cs_final_d3a_emul/
    out = info["output"].rstrip("/")
    oname = os.path.basename(out)
    odir = os.path.dirname(out)
    if os.path.basename(odir) == oname:      # output: folder/name/name
        odir = os.path.dirname(odir)
    name = "MIN_" + oname.replace("MCMC_", "")
    folder = os.path.dirname(os.path.abspath(a.yaml))
    covfile = os.path.join(folder, name + ".covmat")
    C = s.cov(sampled)
    np.savetxt(covfile, C, header=" ".join(sampled), comments="# ")

    info["sampler"] = {"minimize": {
        "method": "bobyqa",
        "best_of": a.best_of,
        "covmat": os.path.relpath(covfile),
        "ignore_prior": bool(a.ignore_prior),
        "max_evals": "1e4d",
        "override_bobyqa": {"rhoend": a.rhoend},
    }}
    info["output"] = os.path.join(odir, name, name)
    path = os.path.join(folder, name + ".yaml")
    yaml.safe_dump(info, open(path, "w"), sort_keys=False, width=100)
    print(f"wrote {path}\n      {covfile}\noutput -> {info['output']}")


if __name__ == "__main__":
    main()
