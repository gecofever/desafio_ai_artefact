"""Conversation persistence.

The chat survives a page reload because every turn is written to SQLite, and
the transcript is exportable as Markdown (the case asks for recorded
conversations). Only the text of each turn is stored: the tool trail is a
rendering detail of the interface, and replaying it to the model would send
stale facts and break the provider's message contract.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.db import get_connection


def save_turn(
    session_id: str,
    role: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls)"
            " VALUES (?, ?, ?, ?)",
            (
                session_id,
                role,
                content,
                json.dumps(tool_calls, ensure_ascii=False, default=str) if tool_calls else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_session(session_id: str) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content, tool_calls FROM messages"
            " WHERE session_id = ? ORDER BY message_id",
            (session_id,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    turns: list[dict[str, Any]] = []
    for row in rows:
        turns.append(
            {
                "role": row["role"],
                "content": row["content"],
                "tool_calls": json.loads(row["tool_calls"]) if row["tool_calls"] else [],
            }
        )
    return turns


def to_markdown(turns: list[dict[str, Any]], title: str = "Conversa") -> str:
    lines = [f"# {title}", ""]
    for turn in turns:
        speaker = "Cliente" if turn["role"] == "user" else "Atendente"
        lines.append(f"**{speaker}:** {turn['content']}")
        for call in turn.get("tool_calls") or []:
            lines.append(
                f"> `{call['name']}({json.dumps(call.get('arguments', {}), ensure_ascii=False)})`"
            )
        lines.append("")
    return "\n".join(lines)
