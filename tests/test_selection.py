from math import sqrt

import numpy as np
import pandas as pd
import pytest

from quant_lab.selection import (
    audit_family,
    bootstrap_means,
    cscv_pbo,
    deflated_sharpe,
    expected_max_sharpe,
    hac_mean,
    matrix,
    returns_to_losses,
    spa_mcs,
)


def panel(n=160, k=4):
    return pd.DataFrame(
        np.random.default_rng(27).normal(0, 0.01, (n, k)),
        index=pd.date_range("2020-01-01", periods=n),
        columns=[f"trial-{i}" for i in range(k)],
    )


def basis():
    return {
        "currency": "USD",
        "benchmark_id": "cash",
        "cost_policy": "net-fees-v1",
        "periods_per_year": 252,
        "sample_kind": "retrospective",
    }


def test_sharpe_units_and_independent_hand_formula():
    data = panel()
    result = deflated_sharpe(data, periods_per_year=252)
    monthly = deflated_sharpe(data, periods_per_year=12)
    first = result["candidates"][0]
    assert first["sensitivity"] == monthly["candidates"][0]["sensitivity"]
    assert first["sharpe_annualized"] == pytest.approx(first["sharpe_period"] * sqrt(252))
    assert first["sharpe_period"] == pytest.approx(data.iloc[:, 0].mean() / data.iloc[:, 0].std())
    assert expected_max_sharpe(0.3, 1) == 0
    assert expected_max_sharpe(0.3, 2) == pytest.approx(0.3 / sqrt(np.pi))
    hurdles = [expected_max_sharpe(0.3, count) for count in (1, 1.5, 2, 3, 10, 100)]
    assert hurdles == sorted(hurdles)
    assert result["candidate_count"] == 4
    assert result["correlations"]["trial-0"]["trial-0"] == 1


def test_noise_search_reveals_raw_winner_inflation_not_automatic_evidence():
    raw, corrected = [], []
    for seed in range(20):
        data = panel(n=200, k=64)
        data[:] = np.random.default_rng(seed).normal(0, 0.01, data.shape)
        result = deflated_sharpe(data, periods_per_year=252)
        winner = max(result["candidates"], key=lambda r: r["sharpe_period"])
        assert (
            data.mean().div(data.std()).max()
            >= data.iloc[:, :4].mean().div(data.iloc[:, :4].std()).max()
        )
        raw.append(winner["psr_zero"])
        corrected.append(winner["sensitivity"][-1]["dsr"])
    assert np.mean(raw) > 0.95
    assert np.mean(corrected) < np.mean(raw) - 0.2


def test_cscv_audit_splits_signal_ties_and_permutation():
    data = panel()
    noise = cscv_pbo(data, blocks=4)
    assert len(noise["splits"]) == 6
    assert noise == cscv_pbo(data[data.columns[::-1]], blocks=4)
    for split in noise["splits"]:
        assert not set(split["train_indices"]) & set(split["test_indices"])
        assert len(split["train_indices"]) == len(split["test_indices"]) == 80
    # A weak but persistent mean advantage relative to each period's noise.
    data["signal"] = data["trial-0"] + 0.006
    assert cscv_pbo(data, blocks=4, metric="mean")["pbo"] < noise["pbo"]
    identical = pd.concat([data["trial-0"]] * 2, axis=1)
    identical.columns = ["a", "b"]
    assert cscv_pbo(identical, blocks=4)["pbo"] == 0.5
    with pytest.raises(ValueError, match="equal-sized"):
        cscv_pbo(data.iloc[:-1], blocks=4)
    with pytest.raises(ValueError, match="excessive"):
        cscv_pbo(data, blocks=4, max_splits=2)


def test_failed_or_missing_candidate_blocks_family_claim():
    data = panel()
    names = list(data.columns) + ["failed-attempt"]
    status = {c: "completed" for c in names}
    status[names[-1]] = "failed"
    audit = audit_family(data, planned=names, statuses=status, family_id="f", basis=basis())
    assert not audit["available"] and audit["unavailable_reasons"][0]["status"] == "failed"
    complete = audit_family(
        data,
        planned=list(data),
        statuses={c: "completed" for c in data},
        family_id="f",
        basis=basis(),
    )
    assert complete["available"]
    data.iloc[1, 1] = np.nan
    assert not audit_family(
        data,
        planned=list(data),
        statuses={c: "completed" for c in data},
        family_id="f",
        basis=basis(),
    )["available"]


def test_loss_sign_spa_mcs_arch_and_column_permutation():
    from arch.bootstrap import SPA

    data = panel()
    benchmark = pd.Series(0.001, index=data.index)
    losses = returns_to_losses(data, benchmark)
    assert losses.iloc[0, 0] == pytest.approx(0.001 - data.iloc[0, 0])
    result = spa_mcs(losses, block_length=5, repetitions=199)
    reference = SPA(np.zeros(len(data)), losses, block_size=5, reps=199, seed=17)
    reference.compute()
    assert result["spa_pvalues"] == reference.pvalues.to_dict()
    assert result == spa_mcs(losses[losses.columns[::-1]], block_length=5, repetitions=199)
    assert set(result["mcs_pvalues"]) == set(data.columns)
    with pytest.raises(ValueError, match="identical dates"):
        returns_to_losses(data, benchmark.iloc[1:])


def test_bootstrap_common_rows_and_arch_reference():
    from arch.bootstrap import StationaryBootstrap

    data = panel(k=2)
    data.iloc[:, 1] = data.iloc[:, 0] * 2
    result = bootstrap_means(data, block_lengths=[3, 9], repetitions=199)
    assert len(result["sensitivity"]) == 2
    row = result["sensitivity"][0]["candidates"]
    np.testing.assert_allclose(row["trial-1"]["ci_95"], np.array(row["trial-0"]["ci_95"]) * 2)
    bs = StationaryBootstrap(3, data.to_numpy(), seed=17)
    reference = np.array([args[0].mean(axis=0) for args, _ in bs.bootstrap(199)])
    np.testing.assert_allclose(
        row["trial-0"]["ci_95"], np.quantile(reference[:, 0], [0.025, 0.975])
    )
    assert (
        bootstrap_means(data, block_lengths=[3], repetitions=100, method="circular")["method"]
        == "circular"
    )


def test_hac_independent_reference_and_autocorrelated_coverage():
    from arch.utility.cov import cov_nw

    values = panel().iloc[:, 0].to_numpy()
    result = hac_mean(values, lags=4)
    assert result["standard_error"] ** 2 == pytest.approx(cov_nw(values, lags=4) / len(values))
    covered, widths, iid_widths = 0, [], []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        x = np.zeros(1000)
        for i, innovation in enumerate(rng.normal(size=999), start=1):
            x[i] = 0.6 * x[i - 1] + innovation
        x = x[200:]
        evidence = hac_mean(x, lags=20)
        covered += evidence["ci_95"][0] <= 0 <= evidence["ci_95"][1]
        widths.append(evidence["standard_error"])
        iid_widths.append(np.std(x, ddof=1) / sqrt(len(x)))
    assert 85 <= covered <= 100  # seeded finite Monte Carlo smoke, not a universal guarantee
    assert np.mean(widths) > 1.5 * np.mean(iid_widths)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.assign(bad=np.nan),
        lambda x: x.iloc[::-1],
        lambda x: pd.concat([x, x]),
        lambda x: x.set_axis(range(len(x))),
    ],
)
def test_invalid_matrix_fails_closed(mutation):
    with pytest.raises(ValueError):
        matrix(mutation(panel()))


@pytest.mark.parametrize(
    "kwargs", [{"periods_per_year": 0}, {"periods_per_year": 252, "effective_trials": 100}]
)
def test_bad_dsr_configuration(kwargs):
    with pytest.raises(ValueError):
        deflated_sharpe(panel(), **kwargs)


def test_degenerate_candidates_fail_instead_of_vanishing():
    data = panel().assign(constant=0)
    with pytest.raises(ValueError, match="constant"):
        deflated_sharpe(data, periods_per_year=252)
    with pytest.raises(ValueError, match="constant"):
        spa_mcs(data, block_length=4)
    with pytest.raises(ValueError):
        bootstrap_means(panel(), block_lengths=[1, 1])
    with pytest.raises(ValueError):
        hac_mean([1, np.nan, 2], lags=1)
