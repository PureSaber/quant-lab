# Independent financial benchmarks

Run the actual upstream APIs against scalar Decimal expectations without loading a
shared heavyweight strategy environment:

```
python -m quant_benchmarks --sources sources.json --out acceptance/new-run
quant-lab financial-benchmarks --sources sources.json --out acceptance/another-new-run
```

The sources JSON maps `quant-data-kit`, `quant-execution`, `quant-fund`, and
`quant-report-hub` to `{ "path": "/absolute/clean/checkout", "commit": "full-sha" }`.
`benchmarks/sources.lock.json` contains the reviewed default-branch pins. Source roots
and imported module paths are checked; each HEAD and clean state is verified both before
and after execution. Missing dependencies, dirty checkouts, mismatched pins and adapter
exceptions produce unavailable/failed results and exit 2. Outputs are exclusive; prior
reports and failed attempts are never overwritten. No fetching or installing happens
inside the harness. Use `--case` repeatedly for a declared subset; a subset is not the
full four-case acceptance.

The runner writes the oracle before invoking adapters, then `report.json`, `report.md`,
source/module identities, Python/package versions, harness file hashes, observed values,
residuals, tolerances, negative controls and raw journals/traces. The FX adapter also
runs the real report writer and retains its independently verifiable report bundle.

| Case | Actual API | Independent expected result |
|---|---|---|
| Split, dividend entitlement/payment, priced delist | QExec ExactAccountLedger + ActionTerms | 800 cash + 40 shares × 30; 3-for-1 preserves 2000 NAV; ex mark 9.75 + 30 receivable preserves NAV; terminal cash at 8 leaves 1790 and zero shares |
| Futures settlement, margin and roll | QExec DerivativeAccount | 3 × multiplier 5 × ((83−80)+(84−86)) = 15 P&L; four trade legs charge 24; 1200+15−24 = 1191; margin 225 is encumbered, never expensed |
| FOF fee/interest and performance HWM | quant-fund FOFEconomics | ACT/365F three-day management .36, custody .072, interest .72; net 1200.288. Separate HWM example reverses reserve on loss, crystallizes 10, sets HWM 240, then charges only 3 on 255 |
| FX internal transfer | report-hub portfolio attribution API/report writer | +10 EUR at 1.2 pairs −12 USD; opening 320, closing 389; 26 local translated P&L +40 opening FX +1 flow FX +2 interest =69; external flow zero |

Negative controls reject changed corporate-action IDs, insufficient-margin trades
atomically, unsupported FOF day count, unpaired transfers, missing sources, and deliberately
wrong expected amounts. QExec and FX amounts use exact Decimal equality. The FOF API is
floating point; its independently specified absolute tolerance is **1e-9 major currency
units**, rather than widening tolerances after inspecting outcomes.

These are **synthetic deterministic financial fixtures executed through real software**.
They do not certify real market behavior, strategy effectiveness, exchange-specific rules,
or live trading. Unsupported: unpriced delisting recovery, SPAN/portfolio margin offsets,
FOF investor flows/tax, and causal attribution of reported FX components.

CI runs normal unit/contract gates separately from the pinned cross-repository benchmark
job. Private repository access requires `QUANT_RESEARCH_READ_TOKEN`; absence is an explicit
job failure, never a skipped green financial acceptance.
