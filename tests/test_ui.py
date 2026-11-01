"""UI tests using Streamlit's AppTest framework.

Seeds conversation history directly into st.session_state to avoid calling
the real agent (no network, no token spend). Verifies that the tool trail
is hidden by default and only appears when audit_mode is enabled.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def get_app_with_seed(turns: list[dict], audit_mode: bool = False) -> AppTest:
    """Initialize AppTest with seeded session state, run it, return the app."""
    app_path = Path(__file__).parent.parent / "app" / "ui" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path))

    # Seed session state before running
    at.session_state["turns"] = turns
    at.session_state["audit_mode"] = audit_mode
    at.session_state["session_id"] = "test_session"

    at.run(timeout=30)
    return at


@pytest.fixture
def sample_turns():
    """One user turn + one assistant turn with tool calls."""
    return [
        {"role": "user", "content": "Qual o preço do violão Yamaha?", "tool_calls": []},
        {
            "role": "assistant",
            "content": "O violão Yamaha C40 está disponível por R$ 1.200,00.",
            "tool_calls": [
                {
                    "name": "search_products",
                    "arguments": {"query": "violão Yamaha"},
                    "result": json.dumps(
                        {
                            "products": [
                                {
                                    "name": "Violão Yamaha C40 Clássico",
                                    "price_brl": 1200.00,
                                    "stock_quantity": 5,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "tool_call_id": "call_123",
                }
            ],
        },
    ]


def test_app_renders_without_exception(sample_turns):
    """App starts and renders without errors when given seeded history."""
    at = get_app_with_seed(sample_turns, audit_mode=False)

    # No exceptions should have occurred
    assert not at.exception, f"App raised exception: {at.exception}"

    # Title should be present
    titles = [e.value for e in at.title if e.value == "Atendimento ao cliente"]
    assert len(titles) == 1, "Expected to find the main title"

    # Both messages should be rendered
    markdown_values = [m.value.lower() for m in at.markdown]
    assert any("violão yamaha" in val for val in markdown_values), "User message not found"
    assert any(
        "r$ 1.200" in val or "1.200,00" in val for val in markdown_values
    ), "Assistant response not found"


def test_tool_trail_hidden_by_default(sample_turns):
    """With audit_mode OFF, no tool trail expander should appear."""
    at = get_app_with_seed(sample_turns, audit_mode=False)

    # Check that no expander with trail content exists
    expander_labels = [e.label for e in at.expander]
    trail_labels = [
        lbl for lbl in expander_labels if "Fontes consultadas" in lbl or "consulta(s)" in lbl
    ]

    assert len(trail_labels) == 0, (
        f"Tool trail should be hidden by default, but found expanders: {trail_labels}"
    )

    # Also verify no emoji-based trail markers
    trail_markers = [lbl for lbl in expander_labels if "🔎" in lbl]
    assert len(trail_markers) == 0, f"Old trail format still present: {trail_markers}"


def test_tool_trail_shown_when_audit_on(sample_turns):
    """With audit_mode ON, tool trail expander should appear with correct label."""
    at = get_app_with_seed(sample_turns, audit_mode=True)

    # Should have exactly one expander with the audit label
    expander_labels = [e.label for e in at.expander]
    trail_labels = [lbl for lbl in expander_labels if "Fontes consultadas" in lbl]

    assert len(trail_labels) == 1, (
        f"Expected exactly one trail expander, found: {trail_labels}"
    )

    # Label should match the new format: "Fontes consultadas (N)"
    assert trail_labels[0] == "Fontes consultadas (1)", (
        f"Expected 'Fontes consultadas (1)', got '{trail_labels[0]}'"
    )

    # Verify the expander contains the tool call details
    # The expander should be the last markdown/code block before it
    trail_expander = [e for e in at.expander if "Fontes consultadas" in e.label][0]

    # Check that the expander contains the tool name and arguments
    # (AppTest doesn't expose expander content directly, but we can verify
    # the structure by checking that code blocks exist after the expander label)
    code_blocks = [c.value for c in at.code]
    assert len(code_blocks) >= 2, (
        f"Expected at least 2 code blocks (arguments + result), found {len(code_blocks)}"
    )

    # First code block should be the arguments (JSON with query)
    assert any("violão Yamaha" in cb for cb in code_blocks), (
        "Tool arguments not found in code blocks"
    )

    # Second code block should be the result (JSON with product data)
    assert any("1200" in cb or "1.200" in cb for cb in code_blocks), (
        "Tool result not found in code blocks"
    )


def test_audit_mode_toggle_caption(sample_turns):
    """Caption text changes based on audit_mode."""
    # With audit mode OFF
    at_off = get_app_with_seed(sample_turns, audit_mode=False)
    captions_off = [c.value for c in at_off.caption]

    # Should have the base caption but NOT the trail instruction
    trail_instruction = any("Abra as consultas" in cap for cap in captions_off)

    assert not trail_instruction, "Trail instruction should be hidden when audit_mode is OFF"

    # With audit mode ON
    at_on = get_app_with_seed(sample_turns, audit_mode=True)
    captions_on = [c.value for c in at_on.caption]

    # Should have the trail instruction
    trail_instruction_on = any("Abra as consultas" in cap for cap in captions_on)
    assert trail_instruction_on, "Trail instruction should appear when audit_mode is ON"
