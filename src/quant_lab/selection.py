"""Family-level research evidence, never a probability of future profitability.

DSR: Bailey/Lopez de Prado (2014), equations 1–2. CSCV: Bailey et al.,
Probability of Backtest Overfitting, section 2. All Sharpe calculations use
the observation period; annualization is display-only. Serial dependence
is NOT repaired by the PSR moment adjustment; use block sensitivity too.
"""

from __future__ import annotations

from itertools import combinations
from math import comb, exp, log, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

NORMAL = NormalDist()
EULER = 0.5772156649015329


def matrix(values: pd.DataFrame, *, minimum=4) -> pd.DataFrame:
    """No inner joins, dropna, imputation, or silent candidate removal."""
    if not isinstance(values, pd.DataFrame) or len(values) < minimum or not len(values.columns):
        raise ValueError("a nonempty candidate matrix with enough observations is required")
    if (
        not isinstance(values.index, pd.DatetimeIndex)
        or values.index.hasnans
        or not values.index.is_unique
        or not values.index.is_monotonic_increasing
        or not values.columns.is_unique
        or any(not isinstance(c, str) or not c.strip() for c in values.columns)
    ):
        raise ValueError("ordered unique dates and unique nonempty candidate IDs required")
    result = values.loc[:, sorted(values.columns)].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("incomplete/nonfinite matrix; retain missing candidates in the audit")
    return result


def _sharpe(values):
    sd = values.std(axis=0, ddof=1)
    if (sd <= 0).any() or not np.isfinite(sd).all():
        raise ValueError("Sharpe is undefined for a constant candidate")
    return values.mean(axis=0) / sd


def expected_max_sharpe(trial_sharpe_std: float, independent_trials: float) -> float:
    if not np.isfinite([trial_sharpe_std, independent_trials]).all():
        raise ValueError("finite trial dispersion and trial count required")
    if trial_sharpe_std < 0 or independent_trials < 1:
        raise ValueError("nonnegative dispersion and at least one trial required")
    if independent_trials == 1 or trial_sharpe_std == 0:
        return 0.0
    # The extreme-value approximation is poor for N between 1 and 2. Interpolate
    # to the exact E[max(Z1,Z2)]=1/sqrt(pi), instead of a negative null hurdle.
    if independent_trials <= 2:
        return trial_sharpe_std * (independent_trials - 1) / sqrt(np.pi)

    def approximation(n):
        return (1 - EULER) * NORMAL.inv_cdf(1 - 1 / n) + EULER * NORMAL.inv_cdf(
            1 - 1 / (n * exp(1))
        )

    # Bridge exact N=2 to the paper's N=3 approximation continuously. This is
    # an explicit small-N convention, not an extra claim from the paper.
    if independent_trials < 3:
        weight = independent_trials - 2
        return trial_sharpe_std * ((1 - weight) / sqrt(np.pi) + weight * approximation(3))
    return trial_sharpe_std * approximation(independent_trials)


def deflated_sharpe(returns, *, periods_per_year, effective_trials=None):
    data = matrix(returns)
    if type(periods_per_year) is not int or periods_per_year < 1:
        raise ValueError("explicit positive periods_per_year required")
    x = data.to_numpy()
    sharpes = _sharpe(x)
    n = len(sharpes)
    dispersion = float(np.std(sharpes, ddof=1)) if n > 1 else 0.0
    corr = data.corr()
    mean_corr = float(corr.to_numpy()[np.triu_indices(n, 1)].mean()) if n > 1 else 1.0
    heuristic = float(np.clip(1 + (n - 1) * (1 - mean_corr), 1, n))
    counts = {1.0, float(n), heuristic}
    if effective_trials is not None:
        if not np.isfinite(effective_trials) or not 1 <= effective_trials <= n:
            raise ValueError("effective_trials must be between 1 and observed candidate count")
        counts.add(float(effective_trials))
    rows = []
    for col, name in enumerate(data.columns):
        centered = x[:, col] - x[:, col].mean()
        scale = sqrt(float(np.mean(centered**2)))
        skew = float(np.mean((centered / scale) ** 3))
        kurt = float(np.mean((centered / scale) ** 4))  # Pearson, not excess
        sr = float(sharpes[col])
        variance = 1 - skew * sr + (kurt - 1) * sr**2 / 4
        if variance <= 0 or not np.isfinite(variance):
            raise ValueError("invalid Sharpe sampling variance")
        sensitivity = []
        for count in sorted(counts):
            hurdle = expected_max_sharpe(dispersion, count)
            score = NORMAL.cdf((sr - hurdle) * sqrt(len(data) - 1) / sqrt(variance))
            sensitivity.append({"independent_trials": count, "hurdle_period": hurdle, "dsr": score})
        rows.append(
            {
                "candidate": name,
                "sharpe_period": sr,
                "sharpe_annualized": sr * sqrt(periods_per_year),
                "skew": skew,
                "pearson_kurtosis": kurt,
                "psr_zero": NORMAL.cdf(sr * sqrt(len(data) - 1) / sqrt(variance)),
                "sensitivity": sensitivity,
            }
        )
    return {
        "method": "DSR-moment-adjusted",
        "observations": len(data),
        "candidate_count": n,
        "periods_per_year": periods_per_year,
        "trial_sharpe_std_period": dispersion,
        "correlations": corr.to_dict(),
        "mean_pairwise_correlation": mean_corr,
        "effective_trials_heuristic": heuristic,
        "effective_trials_policy": "correlation heuristic is sensitivity only, not an identified count",
        "candidates": rows,
        "limitations": [
            "serial dependence not corrected by PSR",
            "higher moments are unstable",
            "not a posterior probability of future profit",
        ],
    }


def cscv_pbo(returns, *, blocks=8, metric="sharpe", max_splits=20000):
    data = matrix(returns)
    if type(blocks) is not int or blocks < 2 or blocks % 2 or len(data) % blocks:
        raise ValueError(
            "CSCV requires an even block count and equal-sized blocks; no tail dropping"
        )
    if len(data) // blocks < 2 or len(data.columns) < 2 or comb(blocks, blocks // 2) > max_splits:
        raise ValueError("insufficient observations/candidates or excessive combinations")
    if metric not in {"mean", "sharpe"}:
        raise ValueError("metric must be mean or sharpe")
    partitions = np.split(np.arange(len(data)), blocks)
    records = []
    x = data.to_numpy()
    score = _sharpe if metric == "sharpe" else lambda a: a.mean(axis=0)
    for chosen in combinations(range(blocks), blocks // 2):
        complement = sorted(set(range(blocks)) - set(chosen))
        train, test = (
            np.concatenate([partitions[i] for i in chosen]),
            np.concatenate([partitions[i] for i in complement]),
        )
        train_scores, test_scores = score(x[train]), score(x[test])
        winners = np.flatnonzero(train_scores == train_scores.max())
        ranks = pd.Series(test_scores).rank(method="average").to_numpy()
        outcomes = []
        for winner in winners:
            relative = float(ranks[winner] / (len(data.columns) + 1))
            outcomes.append(
                {
                    "candidate": data.columns[winner],
                    "is_score": float(train_scores[winner]),
                    "oos_score": float(test_scores[winner]),
                    "oos_relative_rank": relative,
                    "logit": log(relative / (1 - relative)),
                    "below_median": 1.0 if relative < 0.5 else 0.5 if relative == 0.5 else 0.0,
                }
            )
        records.append(
            {
                "train_blocks": list(chosen),
                "test_blocks": complement,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "winners": outcomes,
                "overfit_weight": float(np.mean([o["below_median"] for o in outcomes])),
            }
        )
    return {
        "method": "original-CSCV",
        "metric": metric,
        "blocks": blocks,
        "pbo": float(np.mean([r["overfit_weight"] for r in records])),
        "splits": records,
        "tie_policy": "average all IS winners; OOS midrank; median tie contributes one half",
        "estimand": "IS-selection OOS-relative-rank degradation, not chronological deployment",
    }


def hac_mean(values, *, lags):
    """Bartlett/Newey-West long-run variance; denominator T, no finite-sample correction."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("finite one-dimensional observations required")
    if type(lags) is not int or not 0 <= lags < len(x):
        raise ValueError("lags must be a nonnegative integer below sample size")
    residual = x - x.mean()
    variance = float(residual @ residual) / len(x)
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(residual[lag:] @ residual[:-lag]) / len(x)
    se = sqrt(max(variance, 0) / len(x))
    return {
        "mean": float(x.mean()),
        "standard_error": se,
        "lags": lags,
        "ci_95": [
            float(x.mean()) - 1.959963984540054 * se,
            float(x.mean()) + 1.959963984540054 * se,
        ],
        "assumption": "asymptotic weak-dependence normal approximation",
    }


def bootstrap_means(values, *, block_lengths, repetitions=1000, seed=17, method="stationary"):
    """Resample entire date rows jointly, retaining cross-candidate dependence."""
    from arch.bootstrap import CircularBlockBootstrap, StationaryBootstrap

    data = matrix(values, minimum=10)
    lengths = list(block_lengths)
    if not lengths or any(type(b) is not int or not 1 <= b < len(data) for b in lengths):
        raise ValueError("declare one or more valid block lengths")
    if len(set(lengths)) != len(lengths) or type(repetitions) is not int or repetitions < 100:
        raise ValueError("distinct lengths and at least 100 repetitions required")
    if type(seed) is not int or seed < 0 or method not in {"stationary", "circular"}:
        raise ValueError("explicit seed and stationary/circular method required")
    cls = StationaryBootstrap if method == "stationary" else CircularBlockBootstrap
    x = data.to_numpy()
    observed = x.mean(axis=0)
    results = []
    for length in sorted(lengths):
        sampler = cls(length, x, seed=seed)
        boot = np.asarray([args[0].mean(axis=0) for args, _ in sampler.bootstrap(repetitions)])
        interval = np.quantile(boot, [0.025, 0.975], axis=0)
        centered_p = (1 + (np.abs(boot - observed) >= np.abs(observed)).sum(axis=0)) / (
            repetitions + 1
        )
        results.append(
            {
                "block_length": length,
                "candidates": {
                    name: {
                        "mean": float(observed[i]),
                        "ci_95": interval[:, i].tolist(),
                        "centered_two_sided_p": float(centered_p[i]),
                    }
                    for i, name in enumerate(data.columns)
                },
            }
        )
    return {
        "method": method,
        "seed": seed,
        "repetitions": repetitions,
        "sensitivity": results,
        "resampling": "common date rows across every candidate",
        "limitations": "stationary approximation; report all blocks, do not select lowest p-value",
    }


def returns_to_losses(returns, benchmark):
    """Negative simple net excess return: better investment performance = lower loss."""
    data = matrix(returns)
    if not isinstance(benchmark, pd.Series) or not benchmark.index.equals(data.index):
        raise ValueError("benchmark and candidates must have identical dates")
    if (
        not np.isfinite(benchmark.astype(float)).all()
        or (benchmark < -1).any()
        or (data < -1).any().any()
    ):
        raise ValueError("finite net simple returns no less than -1 required")
    return data.rsub(benchmark, axis=0)


def spa_mcs(losses, *, block_length, repetitions=1000, seed=17, size=0.05):
    """arch SPA vs zero excess-loss benchmark; MCS across the whole fixed family."""
    from arch.bootstrap import MCS, SPA

    data = matrix(losses, minimum=10)
    if len(data.columns) < 2 or type(block_length) is not int or not 1 <= block_length < len(data):
        raise ValueError("at least two candidates and a valid block length required")
    if type(repetitions) is not int or repetitions < 100 or type(seed) is not int or seed < 0:
        raise ValueError("at least 100 repetitions and nonnegative seed required")
    if not 0 < size < 1 or not np.isfinite(size):
        raise ValueError("size must be in (0,1)")
    if (data.std() == 0).any() or data.T.duplicated().any():
        raise ValueError("constant/identical losses need explicit degenerate-family handling")
    spa = SPA(np.zeros(len(data)), data, block_size=block_length, reps=repetitions, seed=seed)
    spa.compute()
    mcs = MCS(data, size=size, block_size=block_length, reps=repetitions, seed=seed, method="R")
    mcs.compute()
    if not np.isfinite(spa.pvalues).all() or not np.isfinite(mcs.pvalues.to_numpy()).all():
        raise ValueError("degenerate bootstrap inference; no significance claim")
    return {
        "method": "arch-SPA-MCS",
        "loss": "negative-net-excess-simple-return",
        "candidate_order": list(data.columns),
        "seed": seed,
        "repetitions": repetitions,
        "block_length": block_length,
        "size": size,
        "spa_pvalues": spa.pvalues.to_dict(),
        "mcs_pvalues": mcs.pvalues.iloc[:, 0].to_dict(),
        "mcs_included": sorted(mcs.included),
        "limitations": "no rejection is not evidence that every model is useless",
    }


def audit_family(returns, *, planned, statuses, family_id, basis):
    """A failed/missing candidate makes family inference unavailable, but remains visible."""
    required = {"currency", "benchmark_id", "cost_policy", "periods_per_year", "sample_kind"}
    if not family_id or not isinstance(basis, dict) or set(basis) != required:
        raise ValueError("family ID and complete common measurement basis required")
    if any(v is None or v == "" for v in basis.values()):
        raise ValueError("measurement basis cannot be empty")
    if type(basis["periods_per_year"]) is not int or basis["periods_per_year"] < 1:
        raise ValueError("explicit positive periods_per_year required")
    if not planned or len(set(planned)) != len(planned) or set(statuses) != set(planned):
        raise ValueError("exact preregistered candidate/status inventory required")
    if any(
        s not in {"completed", "failed", "interrupted", "skipped", "running"}
        for s in statuses.values()
    ):
        raise ValueError("unknown candidate status")
    reasons = []
    for name in sorted(planned):
        if statuses[name] != "completed" or name not in returns:
            reasons.append(
                {"candidate": name, "status": statuses[name], "missing": name not in returns}
            )
    if set(returns.columns) - set(planned):
        raise ValueError("unregistered candidate columns")
    try:
        matrix(returns)
    except (ValueError, TypeError) as exc:
        reasons.append({"matrix_error": str(exc)})
    return {
        "family_id": family_id,
        "basis": dict(basis),
        "planned": sorted(planned),
        "statuses": dict(statuses),
        "available": not reasons,
        "unavailable_reasons": reasons,
    }
