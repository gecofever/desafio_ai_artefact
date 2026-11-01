"""SQLite database built from the challenge CSVs."""

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import CSV_PREFIX, DATA_DIR, STORE_DB_PATH
from app.text import fold


def _csv_path(name: str) -> Path:
    return DATA_DIR / f"{CSV_PREFIX}{name}.csv"


def _parse_specs(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _price(value: str) -> float:
    return float(value.replace(",", ".") if "," in value and "." not in value else value)


def _row_count(path: Path) -> int:
    """Line count matching `wc -l` (used by health-check and data tests)."""
    return path.read_text(encoding="utf-8").count("\n")


def rebuild_database(db_path: Path | None = None) -> dict[str, int]:
    """Idempotently rebuild store.db from data/*.csv."""

    target = Path(db_path or STORE_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            name_folded TEXT
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            description TEXT,
            price_brl REAL NOT NULL,
            stock_quantity INTEGER NOT NULL,
            status TEXT NOT NULL,
            specs TEXT,
            created_at TEXT,
            name_folded TEXT,
            description_folded TEXT
        );

        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            city TEXT
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            total_brl REAL NOT NULL,
            payment_method TEXT,
            tracking_code TEXT,
            estimated_delivery TEXT,
            notes TEXT
        );

        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL
        );

        CREATE TABLE promotions (
            promotion_id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            discount_percent REAL NOT NULL,
            description TEXT,
            is_active INTEGER NOT NULL
        );

        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tool_calls TEXT
        );
        CREATE INDEX idx_messages_session ON messages(session_id);
        """
    )

    counts: dict[str, int] = {}

    with open(_csv_path("categories"), newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = [
            (
                int(r["category_id"]),
                _normalize_text(r["name"]),
                _normalize_text(r["description"]),
                fold(r["name"]),
            )
            for r in reader
        ]
    cur.executemany(
        "INSERT INTO categories (category_id, name, description, name_folded)"
        " VALUES (?, ?, ?, ?)",
        rows,
    )
    counts["categories"] = len(rows)

    with open(_csv_path("products"), newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = []
        for r in reader:
            specs = _parse_specs(r.get("specs", "") or "")
            rows.append(
                (
                    int(r["product_id"]),
                    _normalize_text(r["name"]),
                    int(r["category_id"]),
                    _normalize_text(r["description"]),
                    _price(r["price_brl"]),
                    int(r["stock_quantity"]),
                    r["status"].strip().lower(),
                    json.dumps(specs, ensure_ascii=False),
                    r.get("created_at", "").strip() or None,
                    fold(r["name"]),
                    fold(r["description"]),
                )
            )
    cur.executemany(
        """
        INSERT INTO products
        (product_id, name, category_id, description, price_brl, stock_quantity,
         status, specs, created_at, name_folded, description_folded)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    counts["products"] = len(rows)

    with open(_csv_path("customers"), newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = [
            (
                int(r["customer_id"]),
                _normalize_text(r["name"]),
                r["phone"].strip() if r.get("phone") else None,
                r["email"].strip().lower() if r.get("email") else None,
                _normalize_text(r["city"]) if r.get("city") else None,
            )
            for r in reader
        ]
    cur.executemany(
        "INSERT INTO customers (customer_id, name, phone, email, city) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    counts["customers"] = len(rows)

    with open(_csv_path("orders"), newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = []
        for r in reader:
            tracking = (r.get("tracking_code") or "").strip() or None
            estimated = (r.get("estimated_delivery") or "").strip() or None
            notes = (r.get("notes") or "").strip() or None
            rows.append(
                (
                    int(r["order_id"]),
                    int(r["customer_id"]),
                    r["order_date"].strip(),
                    r["status"].strip().lower(),
                    _price(r["total_brl"]),
                    r.get("payment_method", "").strip().lower() or None,
                    tracking,
                    estimated,
                    notes,
                )
            )
    cur.executemany(
        """
        INSERT INTO orders
        (order_id, customer_id, order_date, status, total_brl, payment_method,
         tracking_code, estimated_delivery, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    counts["orders"] = len(rows)

    with open(_csv_path("order_items"), newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = [
            (int(r["order_id"]), int(r["product_id"]), int(r["quantity"]))
            for r in reader
        ]
    cur.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity) VALUES (?, ?, ?)",
        rows,
    )
    counts["order_items"] = len(rows)

    with open(_csv_path("promotions"), newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = [
            (
                int(r["promotion_id"]),
                int(r["product_id"]),
                float(r["discount_percent"]),
                _normalize_text(r.get("description", "")),
                int(r["is_active"]),
            )
            for r in reader
        ]
    cur.executemany(
        """
        INSERT INTO promotions
        (promotion_id, product_id, discount_percent, description, is_active)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    counts["promotions"] = len(rows)

    conn.commit()
    conn.close()

    # Total file line counts, including headers, used by health checks/tests.
    return {
        "products_total": _row_count(_csv_path("products")),
        "customers_total": _row_count(_csv_path("customers")),
        "orders_total": _row_count(_csv_path("orders")),
        "order_items_total": _row_count(_csv_path("order_items")),
        "promotions_total": _row_count(_csv_path("promotions")),
        "categories_total": _row_count(_csv_path("categories")),
    }


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    target = Path(db_path or STORE_DB_PATH)
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def product_specs_to_flat(specs_json: str) -> dict[str, Any]:
    return _parse_specs(specs_json)
