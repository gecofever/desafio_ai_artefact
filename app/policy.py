"""Deterministic policy engine for Emporio da Musica.

All date windows, thresholds, payment rules and freight rules live here.
`today` is injected; functions never read the current date implicitly.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Campo_Grande")

RETURN_DAYS = 7
DEFECT_DAYS = 30
LEGAL_WARRANTY_DAYS = 90

SHIPPING_FREE_THRESHOLD = 500.0
SHIPPING_FEE = 35.0  # manual §5.1: fixed fee under R$ 500 in the Campo Grande metro area

# Brazilian holidays observed by the store (fixed-date, 2025-2026).
# Used only by store_open; other date calculations ignore holidays.
_HOLIDAYS: set[date] = {
    date(2025, 1, 1),
    date(2025, 4, 18),
    date(2025, 4, 21),
    date(2025, 5, 1),
    date(2025, 6, 19),
    date(2025, 9, 7),
    date(2025, 10, 12),
    date(2025, 11, 2),
    date(2025, 11, 15),
    date(2025, 11, 20),
    date(2025, 12, 25),
    date(2026, 1, 1),
    date(2026, 2, 14),
    date(2026, 2, 16),
    date(2026, 2, 17),
    date(2026, 4, 3),
    date(2026, 4, 6),
    date(2026, 5, 1),
    date(2026, 6, 4),
    date(2026, 9, 7),
    date(2026, 10, 12),
    date(2026, 11, 2),
    date(2026, 11, 15),
    date(2026, 11, 20),
    date(2026, 12, 25),
}


def _to_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _diff_days(start: date, end: date) -> int:
    """Whole days elapsed between start (inclusive) and end (exclusive)."""
    return (end - start).days


def return_window(order_date: date | str, channel: str = "online", today: date | str | None = None) -> dict[str, Any]:
    """7-day return window counted from order/delivery date."""
    start = _to_date(order_date)
    today_date = _to_date(today) if today else date.today()
    deadline = start + timedelta(days=RETURN_DAYS)
    days_left = _diff_days(today_date, deadline)
    eligible = days_left > 0

    reason = None
    if not eligible:
        reason = (
            f"O prazo de {RETURN_DAYS} dias para devolução encerrou em "
            f"{deadline:%d/%m/%Y}. Oferecer garantia do fabricante."
        )
    else:
        reason = f"Dentro do prazo de {RETURN_DAYS} dias. Restam {days_left} dias até {deadline:%d/%m/%Y}."

    return {
        "eligible": eligible,
        "days_left": max(days_left, 0),
        "deadline": deadline.isoformat(),
        "channel": channel.lower().strip(),
        "reason": reason,
    }


def defect_window(purchase_date: date | str, today: date | str | None = None) -> dict[str, Any]:
    """30-day exchange window for manufacturing defects."""
    start = _to_date(purchase_date)
    today_date = _to_date(today) if today else date.today()
    deadline = start + timedelta(days=DEFECT_DAYS)
    days_left = _diff_days(today_date, deadline)
    eligible = days_left > 0

    if eligible:
        reason = (
            f"Dentro do prazo de {DEFECT_DAYS} dias para troca por defeito. "
            f"Restam {days_left} dias até {deadline:%d/%m/%Y}."
        )
    else:
        reason = (
            f"Prazo de {DEFECT_DAYS} dias para troca por defeito encerrado em "
            f"{deadline:%d/%m/%Y}. Encaminhar para garantia do fabricante."
        )

    return {
        "eligible": eligible,
        "days_left": max(days_left, 0),
        "deadline": deadline.isoformat(),
        "next_step": "troca_na_loja" if eligible else "garantia_fabricante",
        "reason": reason,
    }


def legal_warranty(received_date: date | str, today: date | str | None = None) -> dict[str, Any]:
    """90-day legal warranty under the Consumer Protection Code."""
    start = _to_date(received_date)
    today_date = _to_date(today) if today else date.today()
    deadline = start + timedelta(days=LEGAL_WARRANTY_DAYS)
    days_left = _diff_days(today_date, deadline)
    eligible = days_left > 0

    return {
        "eligible": eligible,
        "days_left": max(days_left, 0),
        "deadline": deadline.isoformat(),
        "reason": (
            "Garantia legal de 90 dias ainda vigente."
            if eligible
            else "Garantia legal de 90 dias já expirou. Resta a garantia do fabricante, se houver."
        ),
    }


def pix_price(price: float, has_active_promotion: bool) -> dict[str, Any]:
    """PIX discount is 5% only when there is no active promotion (non-cumulative)."""
    if has_active_promotion:
        return {
            "original_price": round(price, 2),
            "final_price": round(price, 2),
            "discount_percent": 0.0,
            "discount_value": 0.0,
            "reason": "Desconto PIX não acumulativo com promoção ativa.",
        }
    discount = round(price * 0.05, 2)
    return {
        "original_price": round(price, 2),
        "final_price": round(price - discount, 2),
        "discount_percent": 5.0,
        "discount_value": discount,
        "reason": "5% de desconto exclusivo para pagamento via PIX.",
    }


def installments(price: float) -> dict[str, Any]:
    """Available installment plans respecting per-installment minimums."""
    price = float(price)
    plans: list[dict[str, Any]] = []

    for n in range(1, 13):
        installment = price / n
        if n <= 3:
            allowed = price >= 50.0
            minimum_note = "compra mínima de R$ 50,00"
        elif n <= 6:
            allowed = installment >= 80.0
            minimum_note = "parcela mínima de R$ 80,00"
        else:
            allowed = installment >= 100.0
            minimum_note = "parcela mínima de R$ 100,00"

        plans.append(
            {
                "installments": n,
                "allowed": allowed,
                "installment_value": round(installment, 2) if allowed else None,
                "total": round(price, 2) if allowed else None,
                "minimum_note": minimum_note,
            }
        )

    allowed_plans = [p["installments"] for p in plans if p["allowed"]]
    return {
        "price": round(price, 2),
        "allowed_plans": allowed_plans,
        "plans": plans,
        "reason": (
            f"Parcelamento disponível em {allowed_plans} vezes."
            if allowed_plans
            else "Valor abaixo do mínimo para parcelamento."
        ),
    }


def shipping_fee(subtotal: float, city: str | None = None) -> dict[str, Any]:
    """Freight is free for subtotal >= 500; otherwise a flat fee is charged."""
    subtotal = float(subtotal)
    free = subtotal >= SHIPPING_FREE_THRESHOLD
    fee = 0.0 if free else SHIPPING_FEE

    city_norm = (city or "Campo Grande").strip()
    eta_days = 3 if city_norm.lower() == "campo grande" else 5

    return {
        "free": free,
        "threshold": SHIPPING_FREE_THRESHOLD,
        "subtotal": round(subtotal, 2),
        "fee": round(fee, 2),
        "carrier_options": [
            {"carrier": "PAC", "eta_days": eta_days, "fee": round(fee, 2)},
            {"carrier": "Sedex", "eta_days": max(eta_days - 1, 1), "fee": round(fee + 15.0, 2) if not free else 0.0},
        ],
        "eta_days": eta_days,
        "reason": (
            "Frete grátis para compras a partir de R$ 500,00."
            if free
            else f"Frete de R$ {fee:.2f} para compras abaixo de R$ {SHIPPING_FREE_THRESHOLD:.2f}."
        ),
    }


def store_open(now_iso: str | datetime | None = None) -> dict[str, Any]:
    """Check if the physical store is open."""
    if now_iso is None:
        now = datetime.now(TZ)
    elif isinstance(now_iso, datetime):
        now = now_iso.astimezone(TZ)
    else:
        now = datetime.fromisoformat(now_iso).astimezone(TZ)

    day = now.date()
    weekday = day.weekday()  # Monday=0
    is_holiday = day in _HOLIDAYS

    if is_holiday or weekday == 6:
        open_now = False
        hours = None
    elif weekday < 5:  # Monday-Friday
        open_now = 9 <= now.hour < 18
        hours = ("09:00", "18:00")
    else:  # Saturday
        open_now = 9 <= now.hour < 13
        hours = ("09:00", "13:00")

    next_open = "Segunda-feira 09:00" if weekday >= 5 or is_holiday else None

    return {
        "open": open_now,
        "timezone": "America/Campo_Grande",
        "local_time": now.isoformat(),
        "hours_today": hours,
        "is_holiday": is_holiday,
        "next_open": next_open,
    }


_ACCESSORY_KEYWORDS = [
    "cordas",
    "palhetas",
    "palheta",
    "cabos",
    "cabo",
    "cases",
    "case",
    "pedais",
    "pedal",
    "amplificadores",
    "amplificador",
    "afinador",
    "afinadores",
    "correia",
    "correias",
    "encordoamento",
    "encordoamentos",
]

_OUT_OF_SCOPE_KEYWORDS = [
    "restaurante",
    "pizza",
    "delivery",
    "passagem",
    "hotel",
    "viagem",
    "médico",
    "remédio",
    "consulta",
    "advogado",
    "contador",
]


def scope_check(text: str) -> dict[str, Any]:
    """Flag accessory requests and clearly out-of-scope intents."""
    lowered = text.lower()
    accessory = any(kw in lowered for kw in _ACCESSORY_KEYWORDS)
    out_of_scope = any(kw in lowered for kw in _OUT_OF_SCOPE_KEYWORDS)

    return {
        "accessory_request": accessory,
        "out_of_scope": out_of_scope,
        "in_scope": not (accessory or out_of_scope),
        "reason": (
            "A loja vende instrumentos musicais; não comercializamos acessórios avulsos."
            if accessory
            else "Só posso ajudar com assuntos relacionados à Empório da Música."
            if out_of_scope
            else "Dentro do escopo de atendimento da loja."
        ),
    }


_NON_ELIGIBLE_KEYWORDS = [
    "customização",
    "customizado",
    "setup personalizado",
    "produto final de mostruário",
    "mostruário",
    "liquidação final",
    "venda final",
    "boquilha",
    "boquilhas",
]


def non_eligible(text: str) -> dict[str, Any]:
    """Flag exclusions: custom setups, final-sale items and hygiene products."""
    lowered = text.lower()
    matched = [kw for kw in _NON_ELIGIBLE_KEYWORDS if kw in lowered]

    return {
        "eligible_for_return": len(matched) == 0,
        "exclusion_found": bool(matched),
        "exclusions": matched,
        "reason": (
            "Item não elegível à devolução: customização, venda final ou higiene."
            if matched
            else "Item elegível para análise de devolução/troca."
        ),
    }


_POLICY_TOPICS = [
    "returns",
    "defect",
    "warranty",
    "payment",
    "installments",
    "shipping",
    "tracking",
    "hours",
    "scope",
    "promotions",
    "privacy",
    "store_info",
    "loyalty",
]


def policy_lookup(
    topic: str,
    today: date | str | None = None,
    order_date: date | str | None = None,
    delivery_date: date | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return structured policy facts for a given topic.

    When ``order_date`` or ``delivery_date`` is supplied for returns/defect/
    warranty, the matching window is applied deterministically and returned
    under ``window``. ``delivery_date`` takes precedence because the policy
    counts from receipt.
    """
    topic = topic.lower().strip()

    if topic not in _POLICY_TOPICS:
        return {
            "topic": topic,
            "error": "unknown_topic",
            "available_topics": _POLICY_TOPICS,
            "reason": "Tópico desconhecido. Escolha um dos tópicos disponíveis.",
        }

    today_date = _to_date(today) if today else date.today()
    window_start = delivery_date or order_date

    facts: dict[str, Any] = {"topic": topic, "today": today_date.isoformat()}
    if order_date:
        facts["order_date"] = _to_date(order_date).isoformat()
    if delivery_date:
        facts["delivery_date"] = _to_date(delivery_date).isoformat()
    if window_start:
        facts["window_start"] = _to_date(window_start).isoformat()

    if topic == "returns":
        facts["window_days"] = RETURN_DAYS
        facts["condition"] = "Produto sem uso, com nota fiscal e embalagem original."
        facts["channels"] = ["online", "loja_fisica"]
        facts["note"] = "Prazo conta-se da data do recebimento."
        if window_start:
            facts["window"] = return_window(window_start, today=today_date)
    elif topic == "defect":
        facts["window_days"] = DEFECT_DAYS
        facts["condition"] = "Defeito de fabricação comprovado."
        facts["next_step"] = "Troca pelo mesmo produto ou devolução; após 30 dias, garantia do fabricante."
        if window_start:
            facts["window"] = defect_window(window_start, today=today_date)
    elif topic == "warranty":
        facts["legal_warranty_days"] = LEGAL_WARRANTY_DAYS
        facts["condition"] = "Vício oculto ou defeito após o prazo de troca."
        facts["note"] = "Garantia legal do CDC; fabricante pode oferecer prazo maior."
        if window_start:
            facts["window"] = legal_warranty(window_start, today=today_date)
    elif topic == "payment":
        facts["methods"] = ["pix", "cartao_credito", "boleto"]
        facts["pix_discount_percent"] = 5.0
        facts["pix_discount_cumulative"] = False
        facts["pix_discount_with_active_promotion"] = 0.0
        facts["note"] = (
            "PIX não acumula com promoções ativas. Havendo promoção ativa, "
            "o PIX não concede desconto algum."
        )
    elif topic == "installments":
        facts["max_installments"] = 12
        facts["rules"] = {
            "1x_3x": "Compra mínima de R$ 50,00.",
            "4x_6x": "Parcela mínima de R$ 80,00.",
            "7x_12x": "Parcela mínima de R$ 100,00.",
        }
    elif topic == "shipping":
        facts["free_threshold"] = SHIPPING_FREE_THRESHOLD
        facts["flat_fee"] = SHIPPING_FEE
        facts["note"] = "Frete grátis para compras a partir de R$ 500,00."
    elif topic == "tracking":
        facts["format"] = "BR + 9 caracteres alfanuméricos + BR"
        facts["example"] = "BRAB1234567BR"
        facts["note"] = "Pedidos pendentes, cancelados ou não enviados não possuem código."
    elif topic == "hours":
        facts["timezone"] = "America/Campo_Grande"
        facts["weekdays"] = {"mon_fri": "09:00 – 18:00", "saturday": "09:00 – 13:00"}
        facts["sunday_holiday"] = "Fechado"
    elif topic == "scope":
        facts["sells"] = "Instrumentos musicais"
        facts["does_not_sell"] = "Acessórios avulsos (cordas, palhetas, cabos, cases, pedais, amplificadores)"
        facts["note"] = "Redirecionar educadamente para produtos fora do escopo."
    elif topic == "promotions":
        facts["active_only"] = True
        facts["is_active_flag"] = "Promoções inativas devem ser reportadas como expiradas."
        facts["note"] = "Nunca combinar desconto PIX com promoção ativa."
    elif topic == "privacy":
        facts["note"] = "Dados do cliente são usados apenas para processar pedidos e atendimento."
    elif topic == "store_info":
        facts["name"] = "Empório da Música"
        facts["slogan"] = "Sua música começa aqui."
        facts["location"] = "Campo Grande/MS"
    elif topic == "loyalty":
        facts["note"] = "Programa de fidelidade em desenvolvimento."

    return facts
