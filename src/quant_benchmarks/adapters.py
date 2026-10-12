"""Finite real repository adapters. Fixtures are synthetic, API calls are real."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D


def _write(path, value):
    path.write_text(json.dumps(value, default=str, indent=2) + "\n", encoding="utf-8")


def corporate_actions(out):
    from quant_data_kit import AssetClass, FixedPoint, InstrumentSpec, MarkPriceEvent
    from quant_data_kit.exceptions import ValidationError
    from quant_data_kit.financial.actions import ActionTerms
    from quant_execution.ledger import ExactAccountLedger

    at = datetime(2025, 1, 2, tzinfo=timezone.utc)

    def fp(value):
        return FixedPoint.from_decimal(D(str(value)), 2)

    spec = InstrumentSpec(
        instrument_id="BENCH:A",
        asset_class=AssetClass.EQUITY,
        product_type="a_share",
        venue="SYNTHETIC",
        native_symbol="A",
        settlement_currency="CNY",
        price_tick=fp(".01"),
        quantity_step=fp("1"),
        contract_multiplier=fp("1"),
        calendar_id="fixture",
        effective_from=at,
        available_at=at,
        metadata={},
    )
    ledger = ExactAccountLedger(
        account_id="benchmark",
        base_currency="CNY",
        instruments={"BENCH:A": spec},
        initial_cash={"CNY": fp(800)},
        opened_at=at,
    )
    ledger.book_opening_position(
        instrument_id="BENCH:A", quantity=fp(40), average_cost=fp(30), acquired_on=at.date()
    )
    observed = {"opening_nav": ledger.snapshot().nav.to_decimal()}

    def action(kind, day, **fields):
        return ActionTerms(
            event_id=kind,
            instrument_id="BENCH:A",
            kind=kind,
            effective_at=(at + timedelta(days=day)).isoformat(),
            available_at=at.isoformat(),
            currency="CNY",
            source="synthetic",
            evidence_id="independent-benchmark-v1",
            **fields,
        )

    split = action("split", 1, ratio="3")
    snap = ledger.apply_corporate_action(split, at=split.effective_at)
    observed.update(
        split_quantity=snap.positions["BENCH:A"].to_decimal(), split_nav=snap.nav.to_decimal()
    )
    stamp = at + timedelta(days=2)
    ledger.mark(
        MarkPriceEvent(
            event_id="ex-mark",
            instrument_id="BENCH:A",
            event_time=stamp,
            received_at=stamp,
            available_at=stamp,
            source="synthetic",
            trading_day=stamp.date(),
            session_id="fixture",
            sequence=1,
            price=fp("9.75"),
        )
    )
    entitlement = action(
        "dividend_entitlement", 2, cash_per_unit=".25", entitlement_date="2025-01-04"
    )
    ledger.apply_corporate_action(entitlement, at=entitlement.effective_at)
    observed.update(
        receivable=ledger.dividend_receivable_balance("CNY"),
        ex_nav=ledger.snapshot().nav.to_decimal(),
    )
    pay = action("dividend_payment", 3, cash_per_unit=".25", entitlement_date="2025-01-04")
    ledger.apply_corporate_action(pay, at=pay.effective_at)
    observed["paid_cash"] = ledger.cash_balance("CNY")
    terminal = action("terminal_cash", 4, cash_per_unit="8")
    final = ledger.apply_corporate_action(terminal, at=terminal.effective_at)
    observed.update(
        final_cash=ledger.cash_balance("CNY"),
        final_nav=final.nav.to_decimal(),
        final_position=final.positions.get("BENCH:A", fp(0)).to_decimal(),
    )
    repeat = ledger.apply_corporate_action(terminal, at=terminal.effective_at)
    controls = {"idempotent_terminal": repeat == final}
    try:
        from dataclasses import replace

        ledger.apply_corporate_action(
            replace(terminal, cash_per_unit="9"), at=terminal.effective_at
        )
        controls["changed_event_rejected"] = False
    except ValidationError:
        controls["changed_event_rejected"] = ledger.snapshot() == final
    _write(out / "journal.json", [asdict(tx) for tx in ledger.transactions])
    return observed, controls


def futures_roll(out):
    from quant_data_kit.derivatives.models import Contract
    from quant_execution.derivatives import DerivativeAccount

    at = datetime(2025, 1, 2, tzinfo=timezone.utc)
    contracts = [
        Contract(
            instrument_id="BENCH:" + m,
            symbol=m,
            product="BENCH",
            venue="SYNTHETIC",
            currency="USD",
            kind="future",
            multiplier="5",
            tick=".01",
            listed_at=at - timedelta(days=10),
            last_trade_at=at + timedelta(days=d),
            expiry=at + timedelta(days=d),
            known_at=at,
            timezone="UTC",
            settlement="cash",
            initial_margin="75",
            maintenance_margin="60",
            rules_source="Synthetic benchmark assumptions",
        )
        for m, d in [("F", 20), ("N", 50)]
    ]
    account = DerivativeAccount(contracts, 1200, at, fee_per_contract=2)
    trace = []
    account.trade_batch([("BENCH:F", 3, 80)], at, "open")
    snap = account.snapshot()
    trace.append(snap)
    observed = {"open_cash": snap["cash"], "open_margin": snap["initial_margin"]}
    at += timedelta(days=1)
    account.settle_future("BENCH:F", 82, at, "settle")
    observed["settled_cash"] = account.snapshot()["cash"]
    trace.append(account.snapshot())
    at += timedelta(days=1)
    account.trade_batch([("BENCH:F", -3, 83), ("BENCH:N", 3, 86)], at, "roll")
    observed["roll_cash"] = account.snapshot()["cash"]
    trace.append(account.snapshot())
    before = account.snapshot()
    try:
        account.trade_batch([("BENCH:N", 100, 86)], at, "too-much-margin")
        rejected = False
    except ValueError:
        rejected = account.snapshot() == before
    at += timedelta(days=1)
    account.settle_future("BENCH:N", 84, at, "settle-next")
    account.trade_batch([("BENCH:N", -3, 84)], at, "close")
    snap = account.snapshot()
    trace.append(snap)
    observed.update(
        final_cash=snap["cash"],
        final_margin=snap["initial_margin"],
        final_position=account.quantity("BENCH:N"),
    )
    account.validate_balance()
    _write(out / "trace.json", trace)
    _write(out / "journal.json", [asdict(tx) for tx in account.ledger.transactions])
    return observed, {"insufficient_margin_atomic_rejection": rejected}


def fof_fees(out):
    import pandas as pd
    from quant_fund.economics import FOFEconomics, FOFPolicy

    policy = FOFPolicy(
        known_at="2025-01-01", management_rate=0.0365, custody_rate=0.0073, cash_interest_rate=0.073
    )
    economics = FOFEconomics(policy, 1200)
    cash = economics.close(
        pd.Timestamp("2025-01-06"),
        {
            "date": pd.Timestamp("2025-01-03"),
            "total_value": 1200,
            "cash": 1200,
            "fof_fee_liability": 0,
        },
        1200,
        1200,
    )
    row = economics.records[-1]
    observed = {
        "management": row["management_accrual"],
        "custody": row["custody_accrual"],
        "interest": row["cash_interest"],
        "fixed_fee_net": cash,
    }
    perf = FOFEconomics(
        FOFPolicy(
            known_at="2025-01-01",
            performance_rate=0.2,
            crystallization_dates=("2025-01-04", "2025-01-05"),
        ),
        200,
    )
    previous = None
    for day, gross in [("02", 230), ("03", 190), ("04", 250), ("05", 255)]:
        date = pd.Timestamp("2025-01-" + day)
        cash = perf.close(date, previous, gross, gross)
        row = perf.records[-1]
        previous = {
            "date": date,
            "total_value": row["net_assets"],
            "cash": cash,
            "fof_fee_liability": perf.liability,
        }
    observed.update(
        reserve_gain=perf.records[0]["performance_reserve"],
        reserve_loss=perf.records[1]["performance_reserve"],
        crystallized=perf.records[2]["performance_crystallized"],
        post_fee_hwm=perf.records[2]["high_water_mark"],
        next_fee=perf.records[3]["performance_crystallized"],
        next_net=perf.records[3]["net_assets"],
    )
    try:
        FOFPolicy(known_at="2025-01-01", day_count="ACT/360")
        rejected = False
    except ValueError:
        rejected = True
    _write(out / "trace.json", economics.records + perf.records)
    return observed, {"unsupported_day_count_rejected": rejected}


def fx_transfer(out):
    from copy import deepcopy

    from quant_report_hub.portfolio_attribution import evaluate_portfolio_attribution, run_report

    start, end, mid = "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", "2025-01-01T12:00:00Z"
    accounts = []
    for name, currency, opening, closing, fx0, fx1, amount, fx, component in [
        ("eur", "EUR", "200", "230", "1.1", "1.3", "10", "1.2", {"price": "20"}),
        ("usd", "USD", "100", "90", "1", "1", "-12", "1", {"cash_interest": "2"}),
    ]:
        accounts.append(
            {
                "id": name,
                "economic_scope": name,
                "asset_class": "cash" if name == "usd" else "equity",
                "currency": currency,
                "evidence_level": "reported_components",
                "source_ref": "synthetic:independent-v1",
                "data_nature": "synthetic",
                "periods": [
                    {
                        "start": start,
                        "end": end,
                        "known_at": end,
                        "opening_nav": opening,
                        "closing_nav": closing,
                        "fx_start": fx0,
                        "fx_end": fx1,
                        "fx_start_known_at": start,
                        "fx_end_known_at": end,
                        "components": component,
                        "flows": [
                            {
                                "flow_id": name,
                                "amount": amount,
                                "fx_rate": fx,
                                "kind": "internal_transfer",
                                "transfer_id": "paired-1",
                                "time": mid,
                                "known_at": mid,
                                "fx_known_at": mid,
                            }
                        ],
                    }
                ],
            }
        )
    spec = {
        "schema": "quant-report-hub.portfolio-attribution/v1",
        "base_currency": "USD",
        "evaluation_as_of": end,
        "accounts": accounts,
        "risk_models": [],
    }
    _write(out / "input.json", spec)
    result = run_report(out / "input.json", out / "report")
    row = result["portfolio"][0]
    eur = result["accounts"][0]["components_base"]
    observed = {
        key: row[key] for key in ("opening_base", "closing_base", "external_flow_base", "pnl_base")
    }
    observed.update(
        eur_local_pnl_translated=eur["price"],
        fx_opening=eur["fx_on_opening_nav"],
        fx_flow=eur["fx_on_flows"],
    )
    bad = deepcopy(spec)
    bad["accounts"][1]["periods"][0]["flows"][0]["amount"] = "-11"
    try:
        evaluate_portfolio_attribution(bad)
        rejected = False
    except ValueError:
        rejected = True
    return observed, {"unpaired_transfer_rejected": rejected}
