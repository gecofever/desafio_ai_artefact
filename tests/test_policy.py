"""Policy engine tests: every number the agent is allowed to say."""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.policy import (
    installments,
    policy_lookup,
    return_window,
    shipping_fee,
    DEFECT_DAYS,
    LEGAL_WARRANTY_DAYS,
    RETURN_DAYS,
)


def test_return_window_is_seven_days_from_delivery():
    window = return_window("2025-12-05", today="2026-01-01")
    assert RETURN_DAYS == 7
    assert window["deadline"] == "2025-12-12"
    assert window["eligible"] is False
    assert window["days_left"] == 0


def test_return_window_counts_from_delivery_not_from_the_order():
    """Anchoring on the order date can close the window before the parcel arrives."""
    from_delivery = return_window("2025-12-05", today="2025-12-01")
    from_order = return_window("2025-11-20", today="2025-12-01")
    assert from_delivery["deadline"] > from_order["deadline"]
    assert from_delivery["eligible"] is True
    assert from_order["eligible"] is False


def test_every_order_in_the_dataset_has_expired_windows():
    """All orders predate today, so the honest answer is refusal plus warranty."""
    facts = policy_lookup("returns", order_date="2026-03-08", delivery_date="2026-03-20")
    assert facts["window"]["eligible"] is False
    assert facts["window"]["deadline"] == "2026-03-27"
    assert facts["window"]["days_left"] == 0


def test_defect_and_warranty_windows_differ():
    facts = policy_lookup("defect", delivery_date="2025-12-05")
    assert facts["window_days"] == DEFECT_DAYS == 30
    warranty = policy_lookup("warranty", delivery_date="2025-12-05")
    assert warranty["legal_warranty_days"] == LEGAL_WARRANTY_DAYS == 90


def test_shipping_fee_matches_the_manual():
    """Manual §5.1: free above R$ 500, fixed R$ 35,00 below it."""
    assert shipping_fee(500.0)["fee"] == 0.0
    assert shipping_fee(499.99)["fee"] == pytest.approx(35.0)
    assert shipping_fee(120.0)["free"] is False


def test_installment_tiers():
    assert installments(40.0)["allowed_plans"] == []

    mid = installments(300.0)
    assert max(mid["allowed_plans"]) == 3

    high = installments(1000.0)
    assert max(high["allowed_plans"]) == 10  # 7x12 needs R$ 100,00 per instalment
    for option in high["plans"]:
        if option["allowed"] and option["installments"] >= 7:
            assert option["installment_value"] >= 100.0


def test_pix_discount_never_stacks_with_a_promotion():
    without_promo = policy_lookup("payment")
    assert without_promo["pix_discount_percent"] == 5.0
    assert without_promo["pix_discount_cumulative"] is False
    assert without_promo["pix_discount_with_active_promotion"] == 0.0


def test_unknown_topic_lists_the_valid_ones():
    facts = policy_lookup("fidelidade-fidelissima")
    assert facts["error"] == "unknown_topic"
    assert "returns" in facts["available_topics"]


def test_window_topics_accept_an_order_id_through_the_tool():
    """The tool resolves the anchor dates; the model never passes a date."""
    from app.tools import policy_check

    payload = json.loads(policy_check.invoke({"topic": "returns", "order_id": 3}))
    assert payload["window"]["eligible"] is False
    assert payload["window"]["deadline"] < date.today().isoformat()
    assert "anchor_note" in payload
