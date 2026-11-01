"""Data-layer invariants.

These tests encode the data traps the dataset contains, so a future edit to the
loader cannot silently "fix" the source of truth or lose the normalised columns
the search depends on.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.db import get_connection

EXPECTED_ROWS = {
    "products": 65,
    "categories": 9,
    "customers": 50,
    "orders": 20,
    "order_items": 22,
    "promotions": 25,
}


@pytest.fixture()
def conn():
    connection = get_connection()
    yield connection
    connection.close()


def test_every_table_was_loaded(conn):
    for table, expected in EXPECTED_ROWS.items():
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert actual == expected, f"{table}: {actual} != {expected}"


def test_stored_total_is_the_source_of_truth(conn):
    """Orders 3 and 20 disagree with the sum of their items.

    The customer paid `total_brl`, so that column wins and no code path may
    recompute a total from items.
    """
    mismatched = conn.execute(
        """
        SELECT o.order_id, o.total_brl, SUM(oi.quantity * p.price_brl) AS items_sum
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN products p ON p.product_id = oi.product_id
        GROUP BY o.order_id
        HAVING ABS(o.total_brl - items_sum) > 0.01
        ORDER BY o.order_id
        """
    ).fetchall()
    assert [row["order_id"] for row in mismatched] == [3, 20]
    assert mismatched[0]["total_brl"] == pytest.approx(3450.0)


def test_sellable_means_active_and_in_stock(conn):
    active_zero_stock = conn.execute(
        "SELECT COUNT(*) FROM products WHERE status = 'active' AND stock_quantity = 0"
    ).fetchone()[0]
    assert active_zero_stock == 1, "the dataset keeps one active product with no stock"

    not_active = conn.execute(
        "SELECT status, COUNT(*) FROM products WHERE status != 'active' GROUP BY status"
    ).fetchall()
    statuses = {row["status"]: row[1] for row in not_active}
    assert statuses.get("discontinued", 0) >= 2
    assert statuses.get("coming_soon", 0) >= 1


def test_only_four_promotions_are_active(conn):
    active, total = conn.execute(
        "SELECT SUM(is_active), COUNT(*) FROM promotions"
    ).fetchone()
    assert (active, total) == (4, 25)


def test_twelve_orders_have_no_tracking_code(conn):
    missing = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE tracking_code IS NULL OR tracking_code = ''"
    ).fetchone()[0]
    assert missing == 12


def test_categories_without_products_exist(conn):
    empty = conn.execute(
        """
        SELECT c.category_id FROM categories c
        LEFT JOIN products p ON p.category_id = c.category_id
        WHERE p.product_id IS NULL ORDER BY c.category_id
        """
    ).fetchall()
    assert [row[0] for row in empty] == [6, 7, 8]


def test_folded_columns_are_populated(conn):
    """Search depends on `name_folded`; a loader change must not drop it."""
    blank = conn.execute(
        "SELECT COUNT(*) FROM products WHERE name_folded IS NULL OR name_folded = ''"
    ).fetchone()[0]
    assert blank == 0

    violoes = conn.execute(
        "SELECT COUNT(*) FROM categories WHERE name_folded = 'violao'"
    ).fetchone()[0]
    assert violoes == 1


def test_orders_are_all_in_the_past():
    """Every order predates today, which is why every window is expired."""
    from datetime import date

    conn = sqlite3.connect("store.db")
    try:
        newest = conn.execute("SELECT MAX(order_date) FROM orders").fetchone()[0]
    finally:
        conn.close()
    assert date.fromisoformat(newest) < date.today()
