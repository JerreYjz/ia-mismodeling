#!/usr/bin/env python3
"""One fit (NLA, k_max = 0.39; TATT, k_max = 0.70), every earlier number and
the cosmolike value under the same definition. Writes analysis/reconcile_explained.csv."""
import numpy as np
rows = [
 ("v1 (first pass): truth NLA-stitched below k_t, fits to raw P_hE+P_EE, python mask (FLAMINGO total) at K=k_max, B modes in", "1.11", "4.20", "n/a: truth and fits differ"),
 ("v2 matched: residual kept only on 0.05<=k<=K, python mask at K, B modes in", "0.98", "1.20", "1.97 / 0.81 (cosmolike mask, no B)"),
 ("v2 full: residual 0.05-17.7, white offset re-fitted, python mask, B modes in", "1.61", "1.45", "4.96 / 3.27 (cosmolike mask, no B)"),
 ("xi_compare on python mask: residual 0.05-17.7, white level = P_BB plateau, B modes in", "4.80", "2.28", "4.96 / 3.27 (cosmolike mask, no B); python with that definition: 6.05 / 3.05"),
 ("xi_compare, common theta >= 15.8', B modes in", "40.2", "19.3", "114 / 36.6 (no B); python no B: 117 / 36.6"),
 ("production mask 0.35/D(z), cosmolike, no B", "-", "-", "8.74 / 2.69"),
]
with open("analysis/reconcile_explained.csv", "w") as fh:
    fh.write("definition,NLA_kmax0.39_earlier,TATT_kmax0.70_earlier,cosmolike_NLA/TATT\n")
    for r in rows:
        fh.write(",".join('"' + x + '"' for x in r) + "\n")
print(open("analysis/reconcile_explained.csv").read())
