# Research gates

`quant_lab.research_gates` separates native engineering evidence (A), an immutable
future research objective (B), and promotion (C). An objective's point estimate
never grants any of these gates. C always requires external authorization and the
existing model-promotion and real-quote cost requirements.

Recipes may include `research_plan` with schema `quant.research-gate-plan/v1`:
`sample_kind` (`retrospective` for exploratory recipes, or `prospective`), the exact
`sessions`, `review_date`, historical paired `calibration` rows, `confidence`,
`power`, `planning_effect`, `block_lengths`, `repetitions`, and `seed`. Each paired
row has `date`, `strategy`, and `benchmark` net simple returns. The original trial
registry timestamp controls future eligibility; reopening a registry cannot
backdate registration. Future sessions must be inside the recipe's holdout.

Supported primary metrics are mean daily excess return, volatility ratio,
zero-target downside deviation ratio, and worst-5%-tail mean-loss ratio. The
benchmark definition and cost assumptions belong to the frozen objective/recipe.
The declared planning alternative must improve beyond the objective boundary.
Required observations use paired circular-block bootstrap standard errors and
square-root-n extrapolation for that margin, confidence and power. This is a
stationarity-dependent planning approximation, not guaranteed power. Degenerate
calibration is rejected. All declared block lengths must support the conclusion.

`evaluate_gates` requires a validated evidence label, every frozen session, the
scheduled review date, sufficient detectability, matching recomputed point
estimates, confidence intervals beyond the objective boundary, and the existing
objective's measured budgets/data/stop conditions. Missing, provisional or unknown
evidence produces `insufficient_evidence`. Retrospective observations produce
`retrospective_only`. No universal observation count or repeated-trial threshold
is introduced. A development-window result from a prospective recipe still cannot
pass B: the actual evidence dates must match its future holdout sessions.

The equity adapter supplies pairs only for a measured matching benchmark identity;
it does not manufacture capacity, turnover or stop-condition evidence. Callers
must measure those conditions before a complete objective can pass. Findings keep
the gate result alongside the independently verified result artifact.
