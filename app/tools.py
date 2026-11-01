"""The tool surface the model is allowed to use.

Design rule: the model never computes and never remembers. Prices, stock, order
status and promotion validity come from SQLite; day windows, thresholds and
percentages come from `app/policy.py`; policy prose and its citation come from
`app/rag.py`. Every tool returns JSON as a string so the transcript shows
exactly what the model saw.

The `@tool` decorator is what LangChain is doing for us here: it derives the
JSON schema from the signature and docstring, so the tool contract and the
prompt cannot drift apart.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from app import queries
from app.policy import policy_lookup
from app.rag import get_index


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


@tool
def search_products(
    query: str | None = None,
    category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    only_available: bool = True,
    limit: int = 8,
) -> str:
    """Search the catalogue by free text, category and/or price range.

    Use it for any question about whether the store sells something and at what
    price. `only_available=True` keeps only products that are actually sellable
    (status active AND stock above zero). Accents and plurals are folded, so
    "violao", "violão" and "violões" all match.
    """
    return _json(
        queries.search_products(
            query=query,
            category=category,
            min_price=min_price,
            max_price=max_price,
            only_available=only_available,
            limit=limit,
        )
    )


@tool
def get_product(product: str) -> str:
    """Fetch one product by id or name, including its raw status and stock.

    Use it when the customer asks about a specific item, or when a search
    returned nothing and you need to know whether the item exists but is
    discontinued, coming soon or out of stock.
    """
    return _json(queries.get_product(product))


@tool
def list_categories() -> str:
    """List every catalogue category, including categories that have no products."""
    return _json(queries.list_categories())


@tool
def find_customer(query: str) -> str:
    """Look up a customer by name, phone, email or id."""
    return _json(queries.find_customer(query))


@tool
def get_order(order_id: int | None = None, customer: str | None = None) -> str:
    """Fetch an order: status, dates, items, payment method and tracking code.

    Prefer `order_id`. `customer` returns that customer's most recent order.
    A null tracking code means the order has not shipped yet - report that
    instead of inventing a code.
    """
    return _json(queries.get_order(order_id=order_id, customer=customer))


@tool
def get_active_promotions(product: str | None = None) -> str:
    """List active promotions, optionally filtered by product.

    Only active promotions are returned. If the customer asks about one that is
    not in this list, it has expired and the current price applies.
    """
    return _json(queries.get_active_promotions(product=product))


# Topics that resolve a date window and therefore must be anchored on an order.
WINDOW_TOPICS = {"returns", "defect", "warranty"}

# The model speaks Portuguese and the engine speaks English. Aliases are
# resolved here, in code, instead of hoping the prompt narrows the vocabulary.
TOPIC_ALIASES = {
    "pix": "payment",
    "pagamento": "payment",
    "pagamentos": "payment",
    "parcelamento": "installments",
    "parcela": "installments",
    "parcelas": "installments",
    "frete": "shipping",
    "entrega": "shipping",
    "entregas": "shipping",
    "troca": "returns",
    "trocas": "returns",
    "devolucao": "returns",
    "devolução": "returns",
    "arrependimento": "returns",
    "defeito": "defect",
    "garantia": "warranty",
    "horario": "hours",
    "horário": "hours",
    "horas": "hours",
    "funcionamento": "hours",
    "rastreio": "tracking",
    "rastreamento": "tracking",
    "promocao": "promotions",
    "promoção": "promotions",
    "promocoes": "promotions",
    "desconto": "promotions",
    "escopo": "scope",
    "acessorios": "scope",
    "privacidade": "privacy",
    "lgpd": "privacy",
    "loja": "store_info",
    "fidelidade": "loyalty",
}


def _canonical_topic(topic: str) -> str:
    key = (topic or "").strip().lower()
    return TOPIC_ALIASES.get(key, key)


@tool
def policy_check(topic: str, order_id: int | None = None, subtotal: float | None = None) -> str:
    """Deterministic policy answer: windows, thresholds, percentages, hours.

    Topics: returns, defect, warranty, payment, installments, shipping,
    tracking, hours, scope, promotions, privacy, store_info, loyalty.

    Pass `order_id` for returns/defect/warranty: the tool reads the order and
    anchors the window on the delivery estimate (falling back to the order
    date) and returns it already resolved, e.g. `{"eligible": false,
    "deadline": "2025-12-29"}`. Pass `subtotal` for shipping and installments.
    """
    topic = _canonical_topic(topic)
    anchor_note = None
    arguments: dict[str, Any] = {"topic": topic}

    if topic in WINDOW_TOPICS and order_id is not None:
        order = queries.get_order(order_id=order_id)
        if order.get("error") or order.get("not_found"):
            return _json({"error": "order_not_found", "order_id": order_id, "hint": order.get("hint")})
        anchor = order.get("estimated_delivery") or order["order_date"]
        anchor_note = (
            "Janela contada a partir da entrega estimada."
            if order.get("estimated_delivery")
            else "Pedido sem data de entrega registrada; janela contada a partir da data do pedido."
        )
        arguments["order_date"] = order["order_date"]
        arguments["delivery_date"] = anchor

    facts = policy_lookup(**arguments)
    if anchor_note:
        facts["anchor_note"] = anchor_note
    if subtotal is not None:
        from app.policy import installments, shipping_fee

        facts["subtotal"] = subtotal
        facts["shipping"] = shipping_fee(subtotal)
        facts["installments"] = installments(subtotal)
    return _json(facts)


@tool
def policy_search(query: str, k: int = 3) -> str:
    """Find the manual sections that answer a policy question, with citations.

    Returns the section id, title and text (`§4.1 Direito de Arrependimento`).
    Use it for wording, conditions, exceptions and anything you will quote, and
    cite the section id in the reply. Numbers still come from `policy_check`.
    """
    index = get_index()
    hits = index.search(query, k=k)
    if not hits:
        return _json({"results": [], "hint": "Nenhuma seção encontrada."})
    return _json({"results": hits})


TOOLS = [
    search_products,
    get_product,
    list_categories,
    find_customer,
    get_order,
    get_active_promotions,
    policy_check,
    policy_search,
]

TOOL_LABELS = {
    "search_products": "Catálogo",
    "get_product": "Produto",
    "list_categories": "Categorias",
    "find_customer": "Cliente",
    "get_order": "Pedido",
    "get_active_promotions": "Promoções",
    "policy_check": "Regra de política",
    "policy_search": "Manual (RAG)",
}

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
