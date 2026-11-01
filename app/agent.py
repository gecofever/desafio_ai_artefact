"""The agent: one model, eight tools, one explicit loop.

LangChain is used for three things it is genuinely good at - prompt templating,
tool-schema generation (see `app/tools.py`) and the ChatOpenAI client with
`bind_tools`. The loop itself is written out in ~25 lines instead of being
hidden inside an executor graph: this is the file a reviewer will read, and a
hand-written loop can be defended line by line.

Nothing here computes. No price, no deadline, no percentage, no stock level is
derived in this module; it is all requested from tools and quoted back.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.config import (
    MAX_TOOL_ROUNDS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from app.policy import TZ
from app.tools import TOOLS, TOOLS_BY_NAME

SYSTEM_PROMPT = """\
You are the virtual assistant of {store_name} ({slogan}), a musical instrument
store in Campo Grande, Brazil. You answer on WhatsApp and on the website.

# Language
Answer in the language of the customer's most recent message. An English
message gets an English answer, a Portuguese message gets a Portuguese answer.
The store is Brazilian; Portuguese is the default only when the customer's
language is not clear.

# Who you are
You are a salesperson who plays music: warm, informal but professional, never
robotic. The store manual describes the tone you must use (§7.1): the customer
should feel they are talking to a friend who understands music. No emojis in
every message, no corporate filler, short messages.
The store only sells musical instruments. It does not sell loose accessories
(strings, picks, cables, cases, pedals, amplifiers) - if asked, say so kindly
and steer the conversation back to instruments (§1, §7.1).

# Today
Today is {today} (America/Campo_Grande).

# The rule that matters most
You never know a price, a stock level, an order status, a tracking code, a
deadline or a percentage from memory. If a customer asks about any of those,
you call a tool first and answer from its result. If the tool returns nothing,
you say you could not find it - you never fill the gap with a plausible number,
date or code.

Three kinds of questions, three different sources:
1. **Catalogue, orders, customers, promotions** -> `search_products`,
   `get_product`, `list_categories`, `find_customer`, `get_order`,
   `get_active_promotions`.
2. **Rules with numbers** (return window, defect window, warranty, instalments,
   shipping, opening hours, PIX discount) -> `policy_check`. It works with or
   without an order: called with only a topic it returns the rule itself
   (7 days, 30 days, 90 days, 5%, R$ 35,00). Always call it and state the rule
   in the answer, even when the customer has not given an order number yet.
   Pass `order_id` when the question is about a specific order so the window is
   anchored on the real dates. The tool returns the decision already computed
   (`eligible`, `deadline`, `days_left`, `fee`); you phrase it, you do not
   calculate it.
3. **Wording, conditions, exceptions** -> `policy_search` retrieves the manual
   section. Cite the section in the answer, like "(manual, §4.1)".

# Behaviour
- Answer the question that was asked, in the first sentence. Then the detail.
- When a policy window has expired, say the date it closed and offer what is
  still possible (manufacturer warranty, for example). Never invent goodwill.
- An order with no tracking code has not shipped: say that, do not invent a code.
- A promotion that is not in the active list has expired: say so and give the
  current price.
- If a request is outside the store's scope, say so in one line and offer an
  instrument-related alternative. You do not need a tool to state the store's
  own scope (§1).
- If the customer asks about their own order but gives no order number, state
  the rule and what it depends on, then ask for the order number in the same
  message. Never answer with only a request for information.
- Money in Brazilian format (R$ 1.234,56). Never show internal ids unless the
  customer gave one.
"""


def _client() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def build_prompt() -> ChatPromptTemplate:
    from app.config import ASSISTANT_NAME, STORE_SLOGAN

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT.format(
                    store_name=ASSISTANT_NAME,
                    slogan=STORE_SLOGAN,
                    today=datetime.now(TZ).strftime("%Y-%m-%d"),
                ),
            ),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )


def history_to_messages(history: Iterable[dict[str, Any]]) -> list[BaseMessage]:
    """Rebuild the prompt history from stored turns.

    Only text is replayed. Tool calls and tool results from earlier turns are
    deliberately dropped: they are stale (a stock level from three messages ago)
    and they break the provider contract if the stored shape is not exactly
    `tool_calls` + `tool_call_id`. Each turn re-queries what it needs.
    """
    messages: list[BaseMessage] = []
    for turn in history:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def _run_tool(name: str, arguments: dict[str, Any]) -> str:
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return json.dumps({"error": "unknown_tool", "name": name})
    try:
        result = tool.invoke(arguments or {})
    except Exception as exc:  # a broken tool must not break the conversation
        return json.dumps({"error": "tool_failed", "tool": name, "detail": str(exc)})
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def run_turn(
    message: str,
    history: Sequence[dict[str, Any]] | None = None,
    max_rounds: int | None = None,
) -> dict[str, Any]:
    """Answer one customer message. Returns the reply and the tool trail."""
    rounds = max_rounds or MAX_TOOL_ROUNDS
    llm = _client().bind_tools(TOOLS)
    prompt = build_prompt()

    messages: list[BaseMessage] = prompt.format_messages(
        history=history_to_messages(history or []),
        input=message,
    )
    trail: list[dict[str, Any]] = []

    for _ in range(rounds + 1):
        response = llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return {
                "reply": (response.content or "").strip(),
                "tool_calls": trail,
                "rounds": len(trail),
            }

        for call in tool_calls:
            result = _run_tool(call["name"], call.get("args") or {})
            trail.append(
                {
                    "name": call["name"],
                    "arguments": call.get("args") or {},
                    "result": result,
                }
            )
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # Out of rounds: ask for a summary without tools rather than looping forever.
    fallback = _client().invoke(
        messages
        + [
            HumanMessage(
                content=(
                    "Stop using tools now. Answer the customer with what you already "
                    "know, and say plainly if something could not be confirmed."
                )
            )
        ]
    )
    return {
        "reply": (fallback.content or "").strip(),
        "tool_calls": trail,
        "rounds": len(trail),
        "truncated": True,
    }
