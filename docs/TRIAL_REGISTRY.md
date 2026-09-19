# Preregistered trials

`quant_lab.trials.TrialRegistry` adds independent SQLite tables alongside the existing experiment index. The report-hub index schema remains compatible.

Register the hypothesis, explicit allowed parameter sets, code/dependency identity and selection rule before starting a run. The producer calls `start` before fetching data and `finish` on completed or failed attempts. Definitions and event history cannot be updated or deleted through ordinary SQL; registering a changed definition under the same study ID fails. A process killed before `finish` remains visibly running, not successful. After confirming the process has ended, the operator can close its attempt with `finish(id, 'interrupted', {'reason': ...})`.

Prospective holdout dates must start after first registration. The daily strategy freezes its study ID and excludes holdout labels from development diagnostics. `seal_holdout` records one terminal evaluation after the interval ends, against the original code identity and input hash. Re-evaluation under the same study ID is rejected. This is experiment discipline, not a claim that one holdout proves an investment edge.

The existing scanner now indexes blocked decisions without requiring a successful ledger, keeping failed attempts visible in the dashboard. Corrupt standard/v2 remains an error and does not fall back to unverified metrics.
