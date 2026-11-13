# v7: shared modules for the error-budget analysis

These modules are imported by the scripts of `../v8/` (the IA error budget,
`notes/ia_error_budget/`):

* `tolerance_bands.py`: the sim-IA driver setup, the k nodes, the table
  writer and the linear response at a model's best fit.
* `tolerance_nomask.py`: the truth-as-model (`Template`: the injected FLAMINGO
  tables with a free IA amplitude and power-law evolution) and its Jacobian.
* `model_errors.py`: NLA / TATT spectra in the convention of the truth tables
  (`model_ratio`).

The exploratory node-band studies that used to live here were superseded by
`../v8/` and removed (13 November 2026); they remain in the git history
(commit d995c7f and its parent state).
