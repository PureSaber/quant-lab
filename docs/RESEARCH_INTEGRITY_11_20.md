# Research integrity APIs (11–20)

These are research diagnostics, not estimates of future profitability or trading authorization.
No candidate is promoted automatically. Run `pip install -e ".[statistics,dev]"` for the full
statistical suite. `requirements.lock` is the Python 3.10-compatible development closure.

## Register before running (11, 13, 20)

```python
from pathlib import Path
from quant_lab.trials import TrialRegistry
from quant_lab.research import execute_study
from quant_lab.family_evidence import collect_family, evaluate_family

registry = TrialRegistry(Path("research-family.db"))
basis = dict(currency="CNY", benchmark_id="cash-zero-interest-v1",
             cost_policy="CN-fees-v1-net", periods_per_year=252,
             sample_kind="historical-continuous-OOS")
registry.register_family("momentum-v1", {
    "hypothesis": "delayed information diffusion after realistic costs",
    "study_ids": ["momentum-a", "momentum-b"], "basis": basis,
})
# Register every member BEFORE viewing its outcomes. Each validated recipe has:
# family_id="momentum-v1", measurement_basis=basis, and its own immutable study_id.
# execute_study(recipe, output, identity=code_sha, data_identity=snapshot,
#               executor=executor, registry=registry)
values, audit = collect_family(registry, "momentum-v1")
# net_benchmark: explicit same-index, same-currency simple net returns.
# evidence = evaluate_family(values, audit, benchmark=net_benchmark,
#                            blocks=8, block_lengths=(5, 10, 20), seed=17)
```

Definitions and family membership are append-only. Every attempt, including failed retries,
interrupted work and unstarted members, remains visible. A single missing candidate/date or
incompatible currency, costs or interval disables family inference. Result files and all
return artifacts are checked against registered byte hashes. Roots come from each attempt's
start event. A family is **not** complete merely because its best study succeeded.

The measurement basis is a declared economic contract, not an automatic proof that input
prices are correct. The executor currency is checked; the common cost/interval metadata
and exact result manifest are checked during collection. Different benchmark identities,
currencies and fee experiments must not be pooled as if comparable. Family-level DSR and
SPA require a supplied benchmark series; the pipeline records its hash. A risky benchmark
produces a benchmark-relative ratio, not the classic risk-free Sharpe.

## Statistical methods (11–14)

`quant_lab.selection` exposes `deflated_sharpe`, `cscv_pbo`, `returns_to_losses`,
`spa_mcs`, `bootstrap_means` and `hac_mean`. Matrices have unique sorted datetime rows,
unique candidate columns and no missing/nonfinite cells. Input simple net returns use
fractions, not percentages; excess returns are arithmetic return differences.

* DSR uses observation-period Sharpe, Pearson kurtosis and population standardized skew.
  Annualized Sharpe is display-only (`periods_per_year`). Cross-trial Sharpe dispersion
  and the full correlation matrix are reported. Results for N=1, actual N and the heuristic
  `clip(1+(N-1)*(1-mean_correlation),1,N)` are all retained. This heuristic is not an identified
  independent-trial count. Small N is explicit: linear N=1–2 interpolation to the exact
  normal maximum at N=2; linear bridge to the paper approximation at N=3. Serial dependence
  is not fixed by higher-moment correction.
* CSCV uses all equal-block half partitions. Unequal trailing blocks are rejected, not
  silently dropped. All IS-tied winners are averaged; OOS midranks and half-weighted median
  ties are an explicit tie convention. Every split, index, rank and logit is returned.
  This is not chronological OOS. Purged combinations belong in quant-factors separately.
* SPA loss is `benchmark_net_return - candidate_net_return`, with a zero benchmark loss.
  MCS uses the same losses and method R. Candidate names are canonically sorted so the seeded
  result does not depend on input column order. Degenerate constant/duplicate matrices fail
  visibly. SPA rejects a joint null; MCS may retain every model.
* Stationary or circular bootstrap resamples the entire candidate vector with common date
  indices. Every declared block length is shown. HAC is Bartlett/Newey–West mean standard
  error, using a normal approximation. Neither bootstrap nor HAC rescues a short, structurally
  changing sample. Do not pick whichever length gives the smallest p-value.

The tests include hand calculations, noise/weak-signal cases, column permutation, direct
arch comparisons and seeded AR(1) coverage simulations. They test method implementation,
not real-market performance.

## Controlled economic experiments (16)

`counterfactuals.intervention_plan(base, changes, benchmarks=...)` changes exactly one
subtree for each of signal, allocation, risk_latch, frequency, fees and delay. `validate_plan`
rejects extra changes even if someone recomputes the hash. `attribution(plan, net_returns)`
requires base, every intervention and three distinct benchmark columns: `passive`,
`same_risk_constrained`, `cash`.

Effects are differences in terminal compounded net return versus base. The selected benchmark
gap equals sum(effects) + `unexplained_and_interaction_residual` exactly. The residual includes
omitted effects and is **not** a separately identified causal interaction. The A-share
`run_paired` adapter executes the plan through its real simulated ledger and preserves failures.

## Nested selection and transfer (18)

`nested.nested_selection` accepts `fit(training_frame, recipe)` and
`evaluate(model, evaluation_frame)` callbacks; the score must be scalar and higher-is-better.
Candidate recipes include all direction, window, neutralization and model choices. Fit every
learned transform inside each inner fold; refit only the selected recipe on outer training.
Optional `label_end_column` purges overlapping training labels and rejects inner validation
outcomes not mature before the outer test. Each decision and its inner scores is hashed before
outer evaluation. These are trusted callbacks, not a sandbox: global future-data access is
the caller's responsibility. The API evaluates a selection procedure, not a continuous ledger.

`freeze_transfer`/`validate_transfer` enforce unchanged source recipes and explicit currency,
cost and trading-rule policies. For an actual audit boundary use `register_transfer` with a
TrialRegistry, a future target interval and fixed code identity; after the interval,
`registry.seal_holdout` records one immutable final evaluation. A standalone caller-supplied
timestamp/hash does not prove that a human has never seen target data.

## Investment objective contract (20)

Recipes may carry `objective`; it becomes part of immutable study registration. Required fields:
mechanism, benchmark_id, one primary metric/direction/threshold, capital_range,
max_drawdown, max_turnover, max_cost_rate, min_capacity, data_conditions, stop_conditions.
Metrics: net_excess_return, net_return, sharpe, tracking_error, drawdown_magnitude,
net_cashflow_yield, capital_efficiency. Goals refer to the recipe's fixed evaluation interval;
annualized Sharpe/tracking error need a declared annualization convention. Currency matches
the recipe. Drawdown is a positive magnitude; cost rate is total fees / initial capital;
turnover is gross traded notional / initial capital over that interval. Capacity is native
currency sustainable capital, not a daily volume estimate. Cash-flow yield and capital
efficiency require separately documented numerator/denominator evidence, never inferred P&L.

`evaluate_objective` separates return, risk, cost, execution, data and stop checks. Missing
evidence stays unknown, any failed hard condition fails acceptance, and only all satisfied
conditions yield `meets_objective`. A-share provides observed P&L/drawdown/costs/capital but
does not invent capacity or data/stop-condition approvals. Quant-agent reviews this contract
read-only. Missing objectives cannot be retrofitted after seeing results.

## Financial invariants (19)

`invariants` checks signed cash conservation per currency, duplicate event IDs, split wealth,
base-per-local FX direction, future-prefix stability and cost monotonicity for **identical fixed
orders only**. Decimal hand-ledger comparisons and deliberate faults are regression oracles.
No claim is made of a LEAN integration; differing engine assumptions must be reconciled first.

## Primary method references

* [DSR paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
* [PBO paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
* [arch SPA](https://bashtage.github.io/arch/multiple-comparison/generated/arch.bootstrap.SPA.html)
* [arch MCS](https://bashtage.github.io/arch/multiple-comparison/generated/arch.bootstrap.MCS.html)
* [arch stationary bootstrap](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html)
