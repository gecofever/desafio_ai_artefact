"""Agent tests.

The history test is a regression guard: replaying the tool trail from a previous
turn is exactly the bug that makes the second message of a conversation fail,
so it is asserted without needing a model.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import _run_tool, history_to_messages, run_turn


def test_history_replays_text_only():
    history = [
        {"role": "user", "content": "Quanto custa o violão?"},
        {
            "role": "assistant",
            "content": "O C40 sai por R$ 599,90.",
            "tool_calls": [
                {
                    "name": "search_products",
                    "arguments": {"query": "violão"},
                    "result": '{"results": []}',
                }
            ],
        },
        {"role": "user", "content": "E o C70?"},
    ]

    messages = history_to_messages(history)

    assert [type(m) for m in messages] == [HumanMessage, AIMessage, HumanMessage]
    for message in messages:
        assert not isinstance(message, ToolMessage)
        assert not getattr(message, "tool_calls", None)
        assert message.content


def test_empty_history_turns_are_skipped():
    assert history_to_messages([{"role": "assistant", "content": "   "}]) == []


def test_unknown_tool_is_reported_not_raised():
    payload = _run_tool("does_not_exist", {})
    assert "unknown_tool" in payload


def test_tool_failure_is_reported_not_raised():
    payload = _run_tool("get_order", {"order_id": "not-a-number"})
    assert "error" in payload


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_two_turns_in_one_conversation():
    """The path a unit test with a mocked client never touches."""
    first = run_turn("Vocês têm violão Yamaha em estoque? Quanto custa?")
    assert first["tool_calls"], "expected at least one tool call"
    assert "R$" in first["reply"]

    history = [
        {"role": "user", "content": "Vocês têm violão Yamaha em estoque? Quanto custa?"},
        {"role": "assistant", "content": first["reply"]},
    ]
    second = run_turn("E aceita PIX nesse valor?", history)
    assert second["reply"]
    assert "problema interno" not in second["reply"].lower()
    assert "não consegui" not in second["reply"].lower()
