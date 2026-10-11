"""Independent Decimal expectations, specified before calling any production adapter.

Amounts use major currency units; returns are decimal fractions. Each equation is
scalar arithmetic on fixture terms, never production functions or production NAV.
"""

from decimal import Decimal as D


def expectations():
    quantity = D(40) * 3
    dividend = quantity * D(".25")
    corp = {
        "opening_nav": D(800) + 40 * D(30),
        "split_quantity": quantity,
        "split_nav": D(800) + quantity * 10,
        "receivable": dividend,
        "ex_nav": D(800) + quantity * D("9.75") + dividend,
        "paid_cash": D(800) + dividend,
        "final_cash": D(800) + dividend + quantity * 8,
        "final_nav": D(800) + dividend + quantity * 8,
        "final_position": D(0),
    }
    futures = {
        "open_cash": D(1200) - 3 * 2,
        "open_margin": D(3) * 75,
        "settled_cash": D(1200) - 6 + 3 * 5 * (82 - 80),
        "roll_cash": D(1200) + 3 * 5 * (83 - 80) - 3 * 2 * 3,
        "final_cash": D(1200) + 3 * 5 * ((83 - 80) + (84 - 86)) - 4 * 3 * 2,
        "final_margin": D(0),
        "final_position": D(0),
    }
    fof = {
        "management": D(1200) * D(".0365") * 3 / 365,
        "custody": D(1200) * D(".0073") * 3 / 365,
        "interest": D(1200) * D(".073") * 3 / 365,
        "fixed_fee_net": D(1200) * (1 + (D(".073") - D(".0365") - D(".0073")) * 3 / 365),
        "reserve_gain": (D(230) - 200) * D(".2"),
        "reserve_loss": D(0),
        "crystallized": (D(250) - 200) * D(".2"),
        "post_fee_hwm": D(240),
        "next_fee": (D(255) - 240) * D(".2"),
        "next_net": D(255) - (D(255) - 240) * D(".2"),
    }
    fx = {
        "opening_base": D(200) * D("1.1") + 100,
        "closing_base": D(230) * D("1.3") + 90,
        "external_flow_base": D(0),
        "eur_local_pnl_translated": (D(230) - 200 - 10) * D("1.3"),
        "fx_opening": D(200) * (D("1.3") - D("1.1")),
        "fx_flow": D(10) * (D("1.3") - D("1.2")),
        "pnl_base": D(230) * D("1.3") + 90 - (D(200) * D("1.1") + 100),
    }
    return {
        name: {key: str(value) for key, value in values.items()}
        for name, values in [
            ("corporate_actions", corp),
            ("futures_roll", futures),
            ("fof_fees", fof),
            ("fx_transfer", fx),
        ]
    }


DERIVATIONS = {
    "corporate_actions": "800 cash + 40 shares x 30 = 2000; 3-for-1 -> 120 x 10. Ex price 9.75 and receivable 120 x .25 = 30 preserve NAV. Pay moves receivable to cash. Terminal settlement 8 retires all 120 shares: 800+30+960=1790.",
    "futures_roll": "Three contracts, multiplier 5. P&L = 3*5*((83-80)+(84-86)) = 15. Four legs cost 3*2 each = 24. Final cash 1200+15-24=1191. Roll spread is not cash notional payment. Margin 3*75=225 is encumbered, not an expense.",
    "fof_fees": "ACT/365F Friday-to-Monday gap is 3 calendar days: management .36, custody .072, interest .72, closing 1200.288. Separate performance example: HWM 200, 230 reserves 6; 190 reverses to zero; 250 crystallizes 10 and sets post-fee HWM 240; next 255 charges 3, net 252.",
    "fx_transfer": "USD-funded internal EUR transfer: +10 EUR at 1.2 pairs with -12 USD. EUR 200->230 gives local P&L 20, translated 26; opening FX 40; flow FX 1; cash interest 2. Base opening 320, closing 389, P&L 69, external flow 0.",
}
