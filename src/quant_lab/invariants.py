"""Independent financial checks for harmonized, fixed event paths."""

from decimal import Decimal

from quant_lab.research import canonical


def amount(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("finite decimal amount required")
    return result


def assert_cash_conservation(opening, closing, flows):
    """Flows are signed evidenced cash amounts, grouped by currency; no implicit FX."""
    expected = {currency: amount(value) for currency, value in opening.items()}
    ids = set()
    for flow in flows:
        if flow["event_id"] in ids:
            raise AssertionError("duplicate cash event; possible double dividend")
        ids.add(flow["event_id"])
        currency = flow["currency"]
        expected[currency] = expected.get(currency, Decimal(0)) + amount(flow["amount"])
    for currency in set(expected) | set(closing):
        if expected.get(currency, Decimal(0)) != amount(closing.get(currency, 0)):
            raise AssertionError(f"cash conservation failed in {currency}")


def assert_split_wealth(quantity_before, price_before, quantity_after, price_after):
    if amount(quantity_before) * amount(price_before) != amount(quantity_after) * amount(
        price_after
    ):
        raise AssertionError("split changed wealth without another economic event")


def assert_fx_consistency(local_value, quote_base_per_local, base_value):
    if amount(quote_base_per_local) <= 0:
        raise ValueError("positive base-per-local FX rate required")
    if amount(local_value) * amount(quote_base_per_local) != amount(base_value):
        raise AssertionError("FX direction or unit mismatch")


def assert_fixed_order_cost_monotonic(low_cost_nav, high_cost_nav, *, low_orders, high_orders):
    if canonical(low_orders) != canonical(high_orders):
        raise ValueError("cost monotonicity only applies to identical fixed order paths")
    if amount(high_cost_nav) > amount(low_cost_nav):
        raise AssertionError("higher fixed-path costs improved net NAV")


def assert_prefix_invariant(before, after):
    if canonical(before) != canonical(after):
        raise AssertionError("future perturbation changed a past result")
