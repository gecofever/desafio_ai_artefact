"""Streamlit chat interface.

Run with:

    streamlit run app/ui/streamlit_app.py

The chat stays customer-facing by default; the tool trail is an audit tool,
not chat furniture, so it is hidden behind a sidebar toggle ("Ver fontes
consultadas — modo auditoria"). Turn it on to inspect what each answer
looked up; leave it off for a clean conversation.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent import run_turn  # noqa: E402
from app.config import (  # noqa: E402
    OPENAI_MODEL,
    STORE_DB_PATH,
    ASSISTANT_NAME,
    STORE_SLOGAN,
)
from app.sessions import load_session, save_turn, to_markdown  # noqa: E402
from app.tools import TOOL_LABELS  # noqa: E402

SAMPLE_QUESTIONS = [
    "Vocês têm violão Yamaha em estoque? Quanto custa?",
    "Qual o frete para Campo Grande?",
    "Comprei no PIX, tem desconto? E quantas vezes posso parcelar?",
    "Meu pedido 3 atrasou, onde está?",
    "Comprei um violão há 2 meses e veio com defeito. Posso trocar?",
    "Vocês vendem cordas e palhetas?",
]

st.set_page_config(
    page_title=f"{ASSISTANT_NAME} — atendimento",
    page_icon="🎸",
    layout="wide",
)


def _init_state() -> None:
    query_session = st.query_params.get("s")
    if "session_id" not in st.session_state:
        st.session_state.session_id = query_session or uuid.uuid4().hex[:12]
        st.query_params["s"] = st.session_state.session_id
    if "turns" not in st.session_state:
        st.session_state.turns = load_session(st.session_state.session_id)
    if "audit_mode" not in st.session_state:
        st.session_state.audit_mode = False


def _render_trail(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    if not st.session_state.get("audit_mode"):
        return
    with st.expander(f"Fontes consultadas ({len(tool_calls)})"):
        for index, call in enumerate(tool_calls, start=1):
            label = TOOL_LABELS.get(call["name"], call["name"])
            st.markdown(f"**{index}. {label}** · `{call['name']}`")
            st.code(
                json.dumps(call.get("arguments", {}), ensure_ascii=False, indent=2),
                language="json",
            )
            try:
                parsed = json.loads(call["result"])
                st.code(json.dumps(parsed, ensure_ascii=False, indent=2)[:4000], language="json")
            except (json.JSONDecodeError, TypeError):
                st.code(str(call["result"])[:4000], language="text")


_init_state()

with st.sidebar:
    st.header(ASSISTANT_NAME)
    st.caption(f"{STORE_SLOGAN} · Campo Grande/MS")

    st.toggle(
        "Ver fontes consultadas (modo auditoria)",
        value=False,
        key="audit_mode",
        help="Mostra, sob cada resposta, a trilha de consultas feitas pelo atendente.",
    )

    st.markdown(
        "Atendimento sobre **catálogo, pedidos e políticas**. "
        "Preços, estoque e prazos são consultados no sistema a cada pergunta."
    )
    if not STORE_DB_PATH.exists():
        st.error(f"Banco não encontrado em `{STORE_DB_PATH}`. Rode `python scripts/build_db.py`.")

    st.divider()
    st.subheader("Perguntas de exemplo")
    for question in SAMPLE_QUESTIONS:
        if st.button(question, use_container_width=True, key=f"q-{question}"):
            st.session_state.pending = question

    st.divider()
    st.caption(f"Sessão `{st.session_state.session_id}` · modelo `{OPENAI_MODEL}`")
    if st.session_state.turns:
        st.download_button(
            "Exportar conversa (.md)",
            data=to_markdown(st.session_state.turns),
            file_name=f"conversa-{st.session_state.session_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    if st.button("Nova conversa", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.turns = []
        st.query_params["s"] = st.session_state.session_id
        st.rerun()

st.title("Atendimento ao cliente")
if st.session_state.get("audit_mode"):
    st.caption(
        "O atendente consulta catálogo, pedidos e o manual antes de responder. "
        "Abra as consultas abaixo de cada resposta para ver a origem dos dados."
    )
else:
    st.caption(
        "O atendente consulta catálogo, pedidos e o manual antes de responder."
    )

for turn in st.session_state.turns:
    with st.chat_message("user" if turn["role"] == "user" else "assistant"):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            _render_trail(turn.get("tool_calls") or [])

pending = st.session_state.pop("pending", None) if "pending" in st.session_state else None
message = st.chat_input("Escreva sua mensagem para a loja…") or pending

if message:
    with st.chat_message("user"):
        st.markdown(message)

    history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.turns]
    with st.chat_message("assistant"):
        with st.spinner("Consultando o sistema…"):
            try:
                result = run_turn(message, history)
            except Exception as exc:  # never leave the user with a blank screen
                result = {
                    "reply": f"Não consegui responder agora ({exc}). Tente novamente.",
                    "tool_calls": [],
                }
        st.markdown(result["reply"])
        _render_trail(result["tool_calls"])

    st.session_state.turns.append({"role": "user", "content": message, "tool_calls": []})
    st.session_state.turns.append(
        {
            "role": "assistant",
            "content": result["reply"],
            "tool_calls": result["tool_calls"],
        }
    )
    save_turn(st.session_state.session_id, "user", message)
    save_turn(
        st.session_state.session_id,
        "assistant",
        result["reply"],
        result["tool_calls"],
    )
