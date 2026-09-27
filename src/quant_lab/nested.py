"""Chronological nested selection with training-only callbacks and sealed transfer recipes."""

from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant_lab.research import digest


def nested_selection(
    data,
    candidates,
    *,
    fit,
    evaluate,
    outer_train,
    outer_test,
    inner_train,
    inner_test,
    embargo=1,
    label_end_column=None,
):
    """Fit transforms/models INSIDE each inner fold, then refit the selected recipe.

    ``fit(training_frame, recipe) -> model`` and ``evaluate(model, evaluation_frame)
    -> scalar higher-is-better score`` are trusted, caller-owned functions. They are
    handed copies restricted to the declared interval. They must not access global
    future data. Candidate recipes include direction, neutralization and model choices.
    """
    if (
        not isinstance(data.index, pd.DatetimeIndex)
        or not data.index.is_unique
        or not data.index.is_monotonic_increasing
        or data.index.hasnans
    ):
        raise ValueError("ordered unique observation times required")
    if any(
        type(n) is not int or n < 1
        for n in (outer_train, outer_test, inner_train, inner_test, embargo)
    ):
        raise ValueError("positive chronological window and embargo sizes required")
    if (
        not candidates
        or not all(isinstance(k, str) and k for k in candidates)
        or any(not isinstance(v, dict) or not v for v in candidates.values())
    ):
        raise ValueError("explicit named complete candidate recipes required")
    if inner_train + embargo + inner_test > outer_train:
        raise ValueError("outer training interval cannot contain an inner validation fold")
    ends = None
    if label_end_column:
        ends = pd.DatetimeIndex(data[label_end_column])
        if ends.hasnans or (ends < data.index).any():
            raise ValueError("valid label endpoints required")
    records = []

    def training(start, stop, validation_start):
        rows = data.iloc[start:stop].copy(deep=True)
        if ends is not None:
            rows = rows.loc[ends[start:stop] < data.index[validation_start]]
        if rows.empty:
            raise ValueError("no mature training labels")
        return rows

    for test_start in range(outer_train + embargo, len(data) - outer_test + 1, outer_test):
        train_stop = test_start - embargo
        train_start = train_stop - outer_train
        scores = {name: [] for name in sorted(candidates)}
        inner_records = []
        for val in range(
            train_start + inner_train + embargo, train_stop - inner_test + 1, inner_test
        ):
            inner_stop = val - embargo
            inner_data = training(train_start, inner_stop, val)
            validation = data.iloc[val : val + inner_test].copy(deep=True)
            # Inner validation outcomes must themselves be known before outer selection.
            if ends is not None and (ends[val : val + inner_test] >= data.index[test_start]).any():
                raise ValueError("inner validation labels are not mature before outer test")
            row = {
                "train_end": str(inner_data.index[-1]),
                "validation_start": str(validation.index[0]),
                "validation_end": str(validation.index[-1]),
                "scores": {},
            }
            for name in sorted(candidates):
                model = fit(inner_data.copy(deep=True), deepcopy(candidates[name]))
                score = float(evaluate(model, validation.copy(deep=True)))
                if not np.isfinite(score):
                    raise ValueError("failed/nonfinite candidate; nested family is incomplete")
                scores[name].append(score)
                row["scores"][name] = score
            inner_records.append(row)
        selected = min(scores, key=lambda name: (-float(np.mean(scores[name])), name))
        decision = {
            "outer_train_start": str(data.index[train_start]),
            "outer_train_end": str(data.index[train_stop - 1]),
            "test_start": str(data.index[test_start]),
            "test_end": str(data.index[test_start + outer_test - 1]),
            "candidate": selected,
            "recipe": deepcopy(candidates[selected]),
            "inner_folds": inner_records,
        }
        decision["selection_sha256"] = digest(decision)  # sealed before outer evaluation
        model = fit(training(train_start, train_stop, test_start), deepcopy(candidates[selected]))
        score = float(
            evaluate(model, data.iloc[test_start : test_start + outer_test].copy(deep=True))
        )
        if not np.isfinite(score):
            raise ValueError("outer evaluation is not finite")
        records.append({**decision, "outer_score": score})
    if not records:
        raise ValueError("no complete outer folds")
    return {
        "method": "nested-chronological",
        "folds": records,
        "embargo": embargo,
        "unused_tail": int((data.index > pd.Timestamp(records[-1]["test_end"])).sum()),
        "limitations": "callbacks must not use global future data; historical OOS is not prospective",
    }


def freeze_transfer(
    recipe,
    *,
    source_market,
    target_market,
    registered_at,
    target_start,
    currency_policy,
    cost_policy,
    rule_policy,
):
    registered, start = pd.Timestamp(registered_at), pd.Timestamp(target_start)
    if (
        registered.tz is None
        or start.tz is None
        or registered >= start
        or not recipe
        or source_market == target_market
        or not all(
            isinstance(v, str) and v.strip()
            for v in (source_market, target_market, currency_policy, cost_policy, rule_policy)
        )
    ):
        raise ValueError(
            "ex-ante cross-market recipe and explicit currency/cost/rule policies required"
        )
    result = {
        "recipe": deepcopy(recipe),
        "source_market": source_market,
        "target_market": target_market,
        "registered_at": registered.isoformat(),
        "target_start": start.isoformat(),
        "currency_policy": currency_policy,
        "cost_policy": cost_policy,
        "rule_policy": rule_policy,
    }
    return {**result, "sha256": digest(result)}


def validate_transfer(frozen, recipe, *, target_market, at):
    if digest({k: v for k, v in frozen.items() if k != "sha256"}) != frozen.get("sha256"):
        raise ValueError("transfer contract hash mismatch")
    if recipe != frozen["recipe"] or target_market != frozen["target_market"]:
        raise ValueError("transfer cannot retune the frozen source recipe")
    stamp = pd.Timestamp(at)
    if stamp.tz is None or stamp < pd.Timestamp(frozen["target_start"]):
        raise ValueError("target evaluation precedes the frozen holdout")
    return deepcopy(frozen)


def register_transfer(registry, study_id, definition, recipe, *, target_end, now=None, **policies):
    """Anchor the frozen contract in append-only registration before target data arrives.

    Use TrialRegistry.seal_holdout after the declared period for one final evidence
    record. The registry is an audit boundary, not a sandbox against caller code.
    """
    now = now or datetime.now(timezone.utc)
    frozen = freeze_transfer(recipe, registered_at=now.isoformat(), **policies)
    spec = deepcopy(definition)
    spec.update(
        transfer_holdout=frozen,
        holdout_start=pd.Timestamp(frozen["target_start"]).date().isoformat(),
        holdout_end=pd.Timestamp(target_end).date().isoformat(),
    )
    registry.register(study_id, spec, now=now)
    return frozen
