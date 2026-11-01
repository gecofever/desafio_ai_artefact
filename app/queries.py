"""Tool implementations exposed to the LLM.

Every tool returns a JSON-serializable dict. Errors are returned as
`{"error": "...", "hint": "..."}` so the model can recover gracefully.
"""

from __future__ import annotations

import json
import sqlite3
from difflib import get_close_matches
from typing import Any

from app.config import STORE_DB_PATH
from app.db import get_connection
from app.policy import policy_lookup
from app.text import fold as _fold


def _format_price(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")


def _available_sql(only_available: bool) -> tuple[str, list[Any]]:
    if only_available:
        return "AND p.status = 'active' AND p.stock_quantity > 0", []
    return "", []


def search_products(
    query: str | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    only_available: bool = True,
    limit: int = 8,
) -> dict[str, Any]:
    """Search products by text, category and/or price range."""
    try:
        conn = get_connection()
        cur = conn.cursor()

        params: list[Any] = []
        where = ["1=1"]
        avail_sql, avail_params = _available_sql(only_available)
        if avail_sql:
            where.append(avail_sql.lstrip("AND "))
            params.extend(avail_params)

        if query:
            # Every token must match somewhere (name, description or category),
            # so "violão Yamaha" finds "Yamaha C40 Nylon Natural" in the Violões
            # category instead of returning nothing.
            tokens = [token for token in _fold(query).split() if token][:6]
            for token in tokens:
                where.append(
                    "(p.name_folded LIKE ? OR p.description_folded LIKE ?"
                    " OR c.name_folded LIKE ?)"
                )
                like = f"%{token}%"
                params.extend([like, like, like])

        if category:
            # Accept either category id or a (possibly accented/plural) name fragment.
            if category.isdigit():
                where.append("p.category_id = ?")
                params.append(int(category))
            else:
                term = _fold(category)
                category_rows = cur.execute(
                    "SELECT category_id, name FROM categories"
                ).fetchall()
                category_ids = [
                    r["category_id"]
                    for r in category_rows
                    if term and term in _fold(r["name"])
                ]
                if category_ids:
                    placeholders = ", ".join("?" for _ in category_ids)
                    where.append(f"p.category_id IN ({placeholders})")
                    params.extend(category_ids)
                else:
                    where.append("1=0")

        if min_price is not None:
            where.append("p.price_brl >= ?")
            params.append(float(min_price))
        if max_price is not None:
            where.append("p.price_brl <= ?")
            params.append(float(max_price))

        sql = f"""
            SELECT p.product_id, p.name, c.name AS category, p.price_brl,
                   p.stock_quantity, p.status,
                   COALESCE(MAX(pr.discount_percent), 0) AS discount_percent
            FROM products p
            JOIN categories c ON c.category_id = p.category_id
            LEFT JOIN promotions pr ON pr.product_id = p.product_id AND pr.is_active = 1
            WHERE {' AND '.join(where)}
            GROUP BY p.product_id
            ORDER BY p.name
            LIMIT ?
        """
        params.append(int(limit))

        rows = cur.execute(sql, params).fetchall()

        results = []
        for r in rows:
            price = float(r["price_brl"])
            discount = float(r["discount_percent"])
            final = round(price * (1 - discount / 100), 2) if discount else price
            results.append(
                {
                    "product_id": r["product_id"],
                    "name": r["name"],
                    "category": r["category"],
                    "price_brl": price,
                    "final_price_brl": final,
                    "discount_percent": discount,
                    "stock_quantity": r["stock_quantity"],
                    "status": r["status"],
                }
            )

        note = "mostrando apenas disponíveis" if only_available else "incluindo indisponíveis"
        if not results:
            note = "Nenhum produto encontrado com os filtros informados."

        return {"results": results, "count": len(results), "note": note}
    except Exception as exc:
        return {"error": "search_failed", "hint": str(exc), "results": [], "count": 0}


def get_product(product: str) -> dict[str, Any]:
    """Return details for a product by numeric id or fuzzy name match."""
    try:
        conn = get_connection()
        cur = conn.cursor()

        by_id: sqlite3.Row | None = None
        if isinstance(product, str) and product.isdigit():
            by_id = cur.execute(
                "SELECT * FROM products WHERE product_id = ?", (int(product),)
            ).fetchone()
        elif isinstance(product, int):
            by_id = cur.execute(
                "SELECT * FROM products WHERE product_id = ?", (product,)
            ).fetchone()

        row = by_id
        if row is None:
            term = _fold(str(product))
            # 1) exact match on the folded name, 2) every token present,
            # 3) give up with suggestions instead of guessing.
            row = cur.execute(
                "SELECT * FROM products WHERE name_folded = ?", (term,)
            ).fetchone()
            if row is None:
                tokens = [token for token in term.split() if token][:5]
                if tokens:
                    clause = " AND ".join(
                        "(name_folded LIKE ? OR description_folded LIKE ?)" for _ in tokens
                    )
                    like_params: list[Any] = []
                    for token in tokens:
                        like_params.extend([f"%{token}%", f"%{token}%"])
                    row = cur.execute(
                        f"SELECT * FROM products WHERE {clause} ORDER BY product_id LIMIT 1",
                        like_params,
                    ).fetchone()
            if row is None:
                rows = cur.execute("SELECT product_id, name_folded, name FROM products").fetchall()
                names = [r["name_folded"] for r in rows]
                matches = get_close_matches(term, names, n=3, cutoff=0.4)
                close = [
                    {"product_id": r["product_id"], "name": r["name"]}
                    for r in rows
                    if r["name_folded"] in matches
                ]
                return {
                    "not_found": True,
                    "query": product,
                    "close_matches": close,
                    "hint": (
                        "Nenhum produto exato encontrado. Tente um dos similares, "
                        "ou use search_products."
                    ),
                }

        category = cur.execute(
            "SELECT name FROM categories WHERE category_id = ?", (row["category_id"],)
        ).fetchone()
        promo = cur.execute(
            "SELECT discount_percent FROM promotions WHERE product_id = ? AND is_active = 1",
            (row["product_id"],),
        ).fetchone()

        specs = json.loads(row["specs"]) if row["specs"] else {}
        price = float(row["price_brl"])
        discount = float(promo["discount_percent"]) if promo else 0.0
        final = round(price * (1 - discount / 100), 2) if discount else price

        return {
            "product_id": row["product_id"],
            "name": row["name"],
            "category": category["name"] if category else None,
            "description": row["description"],
            "price_brl": price,
            "final_price_brl": final,
            "discount_percent": discount,
            "stock_quantity": row["stock_quantity"],
            "status": row["status"],
            "specs": specs,
            "created_at": row["created_at"],
        }
    except Exception as exc:
        return {"error": "lookup_failed", "hint": str(exc)}


def list_categories() -> dict[str, Any]:
    """List all categories and product counts (zero-count categories are expected)."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT c.category_id, c.name,
                   COUNT(p.product_id) AS product_count
            FROM categories c
            LEFT JOIN products p ON p.category_id = c.category_id
            GROUP BY c.category_id
            ORDER BY c.category_id
            """
        ).fetchall()
        return {
            "categories": [
                {
                    "category_id": r["category_id"],
                    "name": r["name"],
                    "product_count": r["product_count"],
                }
                for r in rows
            ]
        }
    except Exception as exc:
        return {"error": "lookup_failed", "hint": str(exc)}


def find_customer(query: str) -> dict[str, Any]:
    """Find a customer by email, phone or name and return recent orders summary."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        term = query.strip().lower()

        # Try exact email first.
        row = cur.execute(
            "SELECT * FROM customers WHERE LOWER(email) = ?", (term,)
        ).fetchone()
        if row is None:
            digits = "".join(ch for ch in term if ch.isdigit())
            if digits:
                row = cur.execute(
                    "SELECT * FROM customers WHERE REPLACE(REPLACE(phone, '(', ''), ')', '') LIKE ?",
                    (f"%{digits}%",),
                ).fetchone()
            if row is None:
                row = cur.execute(
                    "SELECT * FROM customers WHERE LOWER(name) LIKE ? LIMIT 1", (f"%{term}%",)
                ).fetchone()

        if row is None:
            return {
                "not_found": True,
                "query": query,
                "hint": "Nenhum cliente encontrado com esse critério.",
            }

        orders = cur.execute(
            """
            SELECT order_id, order_date, status, total_brl, tracking_code, estimated_delivery
            FROM orders
            WHERE customer_id = ?
            ORDER BY order_date DESC
            LIMIT 5
            """,
            (row["customer_id"],),
        ).fetchall()

        return {
            "customer": {
                "customer_id": row["customer_id"],
                "name": row["name"],
                "phone": row["phone"],
                "email": row["email"],
                "city": row["city"],
            },
            "recent_orders": [
                {
                    "order_id": o["order_id"],
                    "order_date": o["order_date"],
                    "status": o["status"],
                    "total_brl": float(o["total_brl"]),
                    "tracking_code": o["tracking_code"],
                    "estimated_delivery": o["estimated_delivery"],
                }
                for o in orders
            ],
        }
    except Exception as exc:
        return {"error": "lookup_failed", "hint": str(exc)}


def get_order(order_id: int | None = None, customer: str | None = None) -> dict[str, Any]:
    """Return order status, dates, items and tracking."""
    try:
        conn = get_connection()
        cur = conn.cursor()

        row: sqlite3.Row | None = None
        if order_id is not None:
            row = cur.execute(
                "SELECT * FROM orders WHERE order_id = ?", (int(order_id),)
            ).fetchone()
        elif customer:
            cust = find_customer(customer)
            if "error" in cust or cust.get("not_found"):
                return {
                    "error": "customer_not_found",
                    "hint": cust.get("hint", "Cliente não encontrado."),
                }
            cid = cust["customer"]["customer_id"]
            row = cur.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_date DESC LIMIT 1",
                (cid,),
            ).fetchone()
        else:
            return {"error": "missing_parameter", "hint": "Informe order_id ou customer."}

        if row is None:
            return {"not_found": True, "hint": "Pedido não encontrado."}

        items = cur.execute(
            """
            SELECT oi.quantity, p.product_id, p.name, p.price_brl
            FROM order_items oi
            JOIN products p ON p.product_id = oi.product_id
            WHERE oi.order_id = ?
            """,
            (row["order_id"],),
        ).fetchall()

        customer_row = cur.execute(
            "SELECT name FROM customers WHERE customer_id = ?", (row["customer_id"],)
        ).fetchone()

        return {
            "order_id": row["order_id"],
            "customer_id": row["customer_id"],
            "customer_name": customer_row["name"] if customer_row else None,
            "order_date": row["order_date"],
            "status": row["status"],
            "total_brl": float(row["total_brl"]),
            "payment_method": row["payment_method"],
            "tracking_code": row["tracking_code"],
            "estimated_delivery": row["estimated_delivery"],
            "notes": row["notes"],
            "items": [
                {
                    "product_id": i["product_id"],
                    "name": i["name"],
                    "quantity": i["quantity"],
                    "unit_price_brl": float(i["price_brl"]),
                }
                for i in items
            ],
        }
    except Exception as exc:
        return {"error": "lookup_failed", "hint": str(exc)}


def get_active_promotions(product: str | None = None) -> dict[str, Any]:
    """Return active promotions only; inactive promos are reported as expired."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        params: list[Any] = []
        where = ["p.is_active = 1"]

        if product:
            if isinstance(product, str) and product.isdigit():
                where.append("p.product_id = ?")
                params.append(int(product))
            else:
                where.append("pr.name LIKE ?")
                params.append(f"%{product}%")

        sql = f"""
            SELECT p.promotion_id, p.product_id, pr.name AS product_name,
                   p.discount_percent, p.description
            FROM promotions p
            JOIN products pr ON pr.product_id = p.product_id
            WHERE {' AND '.join(where)}
            ORDER BY p.discount_percent DESC
        """
        rows = cur.execute(sql, params).fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "promotion_id": r["promotion_id"],
                    "product_id": r["product_id"],
                    "product_name": r["product_name"],
                    "discount_percent": float(r["discount_percent"]),
                    "description": r["description"],
                    "expires_note": "Promoção ativa no momento.",
                }
            )

        return {
            "count": len(results),
            "promotions": results,
            "note": (
                f"{len(results)} promoção(ões) ativa(s) encontrada(s)."
                if results
                else "Nenhuma promoção ativa. Promoções inativas estão expiradas."
            ),
        }
    except Exception as exc:
        return {"error": "lookup_failed", "hint": str(exc)}


# Re-export policy_lookup so the tools module owns the tool surface.
__all__ = [
    "search_products",
    "get_product",
    "list_categories",
    "find_customer",
    "get_order",
    "get_active_promotions",
    "policy_lookup",
]
