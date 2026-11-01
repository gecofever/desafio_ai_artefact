"""Tool-level tests: what the model is allowed to see must be correct."""

from __future__ import annotations

import json

import pytest

from app import queries


def test_accent_and_plural_variants_find_the_same_products():
    """'violao', 'violão' and 'violões' must all reach the Violões category."""
    results = {
        variant: queries.search_products(query=variant)["count"]
        for variant in ("violao", "violão", "violões", "violoes")
    }
    assert set(results.values()) == {8}, results


def test_multi_word_query_does_not_require_the_exact_phrase():
    """'violão Yamaha': the word 'violão' lives in the category, not the name."""
    response = queries.search_products(query="violão Yamaha")
    names = [item["name"] for item in response["results"]]
    assert response["count"] == 7
    assert all("Yamaha" in name for name in names)


def test_search_never_returns_unsellable_products_by_default():
    response = queries.search_products(query="violão")
    for item in response["results"]:
        assert item["status"] == "active"
        assert item["stock_quantity"] > 0


def test_discontinued_product_is_visible_when_asked_for_explicitly():
    """Availability is not existence: the agent needs both answers."""
    by_name = queries.search_products(query="SN-7C", only_available=False)
    assert by_name["count"] == 1
    assert by_name["results"][0]["status"] == "discontinued"

    # ... and it disappears from the default, sellable-only view.
    assert queries.search_products(query="SN-7C")["count"] == 0


def test_out_of_scope_accessory_returns_nothing():
    """The store does not sell picks; the agent must say so, not improvise."""
    assert queries.search_products(query="palheta")["count"] == 0


def test_empty_category_is_reported_without_products():
    categories = queries.list_categories()["categories"]
    empty = [c for c in categories if c["product_count"] == 0]
    assert len(empty) == 3


def test_order_without_tracking_returns_null_not_a_code():
    order = queries.get_order(order_id=20)
    assert order["tracking_code"] is None
    assert order["status"] == "cancelled"


def test_order_with_tracking_returns_the_stored_code():
    shipped = [
        queries.get_order(order_id=order_id)
        for order_id in range(1, 21)
        if queries.get_order(order_id=order_id)["tracking_code"]
    ]
    assert shipped, "expected at least one shipped order"
    for order in shipped:
        assert order["tracking_code"].startswith("BR")


def test_active_promotions_exclude_expired_ones():
    response = queries.get_active_promotions()
    assert response["count"] == 4
    for promo in response["promotions"]:
        assert promo["discount_percent"] > 0


def test_get_product_accepts_a_folded_name():
    product = queries.get_product("violao yamaha c40")
    assert product["product_id"] == 81
    assert product["final_price_brl"] <= product["price_brl"]


def test_tool_wrappers_return_json_strings():
    """The model receives a string, and the transcript can show it verbatim."""
    from app.tools import get_active_promotions, policy_check, search_products

    for tool, arguments in (
        (search_products, {"query": "guitarra"}),
        (get_active_promotions, {}),
        (policy_check, {"topic": "shipping", "subtotal": 300}),
    ):
        payload = tool.invoke(arguments)
        assert isinstance(payload, str)
        assert json.loads(payload)
