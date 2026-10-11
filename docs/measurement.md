# Economic measurement and comparison

`quant.measurement/v1` is a closed JSON declaration implemented by the **stdlib-only**
`quant_measurement` package within this distribution. Producers can emit JSON without
installing quant-lab. `pip install --no-deps quant-lab` is sufficient for this module
alone; importing it never imports pandas, Arrow or execution libraries.

See [the complete example](../measurement-example.json). All 26 `basis` fields are
required. `validate_measurement(value)` returns a detached valid declaration or raises
`ValueError`/`TypeError`. `compare_measurements(values, statuses=...)` returns
`comparable`, `reasons` (index, field, code, message), and a stable `basis_id`.

```python
from quant_measurement import compare_measurements, content_id, validate_measurement
result = compare_measurements([left, right], statuses=["completed", "completed"])
assert result["comparable"]  # declared economic compatibility, not market validity
```

The basis fixes currency and monetary units, decimal/percent return units, return
formula, starting capital and capital/return denominator policies, exact UTC bounds,
observation frequency/calendar/timezone/grid, annualization, risk-free rate unit and
compounding, external/internal flow treatment, cost and valuation policy hashes,
benchmark, universe and signal/execution timing. There is **no** resampling, currency
conversion, percent conversion, date intersection, cost replacement or missing-value
fill. An event stream uses `annualization_policy="not_annualized"` and
`annualization_periods=null`. `fixed_periods` requires a positive finite numeric period
count and a non-event frequency. Risk-free rates and starting capital are decimal
strings; the former declares `annual_decimal` or `per_period_decimal` and
`simple`, `effective` or `continuous` compounding.

`evidence_kind`: synthetic, retrospective, historical_pit, prospective_observed.
`sample_kind`: deterministic_fixture, in_sample, out_of_sample, prospective.
Equal synthetic declarations permit arithmetic comparison, never investability ranking.
Missing, legacy, invalid, failed, interrupted or skipped inputs remain visible and block
the **entire requested comparison**. At least two requested inputs are required.
Different code hashes are recorded but do not alone block comparison; market input
hashes must match. Candidate strategy recipes remain in existing frozen run configs.
An old/new market revision is an explicit intervention requiring its own paired analysis,
not an exemption from the same-input rule in this ordinary comparison API.

`content_id(value)` is `sha256:` + SHA256 of UTF-8 JSON serialized with sorted keys,
compact separators, `ensure_ascii=True`, `allow_nan=False`, and no newline. The grid
hash covers an array `[opening_boundary, observation_1, ..., last_observation]` of
strictly increasing UTC ISO timestamps ending `Z`. Representations compare exactly;
no rate/unit/time normalization is performed. Unsupported scope lists compare as sets.

## Native source binding

Store the declaration in `standard/v2/metrics.json` under `measurement`.
`quant_lab.measurement.read_measured_run(path)` first validates standard/v2 and then
reads each consumed artifact once into bytes, checks those bytes against that validated
manifest, and parses the same byte snapshots. A concurrent self-consistent replacement
cannot combine an old declaration with new returns.

- `provenance.input_identity` equals `manifest.dataset_snapshots` exactly; values are
  `sha256:` plus 64 lowercase hexadecimal digits and identify frozen **market inputs**.
- `provenance.code_identity` contains `manifest.code_version`; all values are full
  Git SHA-1/SHA-256 hashes.
- `config.measurement_policies.cost` and `.valuation` hold the producer's actual frozen
  policy objects. Their `content_id`s equal `basis.cost_policy_id` and `.valuation_policy_id`.
  Producers must truthfully bind these to the executed configuration; a hash alone does
  not prove algorithmic equivalence.
- Currency equals the native manifest; the observed grid is the declared opening plus
  the actual `returns.event_time` stream. This bridge supports one nonempty strategy
  stream. Aggregate or select strategies explicitly upstream; it never silently selects.

## Product entrypoints

```
quant-lab --db state/experiments.db compare --project PROJECT RUN_A RUN_B --json
quant-lab --db state/experiments.db compare --project PROJECT LEGACY --view-only
```

The first command exits 2 on unavailable/incompatible inputs and includes reasons.
`--view-only` preserves historical table/JSON display and exits 0 without changing
`comparable=false`. Source corruption suppresses cached values. Native metrics cannot
be overwritten by adjacent, unhashed legacy CSVs. `research.compare_results` uses the
same measurement gate and retains failed candidates in `requested_results`.
Rank eligibility additionally requires a native backtest ledger, nonsynthetic evidence,
and existing investability/universe tags; compatibility itself is not a ranking.
