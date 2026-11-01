"""Evaluation harness: a golden set run against the live agent.

    python scripts/eval.py              # agent golden set (needs OPENAI_API_KEY)
    python scripts/eval.py --retrieval  # retrieval only (cheap, no agent)

Why this exists: the case is a prototype, but a prototype you cannot measure is
a demo. Each case declares the tools that must be called and the strings the
answer must (or must not) contain, so a prompt change is judged by evidence
instead of vibes. It is deliberately small and readable - the value is the
habit, not the count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import PolicyIndex  # noqa: E402

RETRIEVAL_CASES = {
    "posso devolver um violão que comprei semana passada?": "4.1",
    "qual o frete para Campo Grande?": "5.1",
    "aceita PIX com desconto?": "3",
    "não recebi o código de rastreio": "5.3",
    "que horas vocês abrem no sábado?": "2",
    "meu violão estragou, e agora?": "4.2",
    "quantas vezes posso parcelar?": "3.1",
    "qual a garantia do fabricante?": "8.2",
    "posso trocar por outro modelo?": "4.3",
    "como vocês enviam para outra cidade?": "5.2",
    "o que a garantia não cobre?": "8.3",
    "vocês vendem palhetas?": ("1", "7.1"),
    "posso combinar PIX com promoção?": "6.2",
}

# Each case: message, tools that must appear, substrings that must be present,
# substrings that must not appear (the traps).
AGENT_CASES: list[dict[str, Any]] = [
    {
        "id": "price-and-stock",
        "message": "Vocês têm violão Yamaha em estoque? Quanto custa o C40?",
        "tools": {"search_products"},
        "must": ["599"],
        "must_not": ["não temos"],
    },
    {
        "id": "pix-not-cumulative",
        "message": "Comprei um violão de R$ 599,90 no PIX. Quanto fica?",
        "tools": {"policy_check"},
        "must": ["5%"],
        "must_not": [],
    },
    {
        "id": "installments",
        "message": "Esse violão de R$ 599,90 parcelado em 6x dá quanto por parcela?",
        "tools": {"policy_check"},
        "must": ["99"],
        "must_not": [],
    },
    {
        "id": "shipping-free-above-500",
        "message": "Se eu comprar um violão de R$ 599,90, o frete para Campo Grande é grátis?",
        "tools": {"policy_check"},
        "must": ["grátis"],
        "must_not": ["R$ 35"],
    },
    {
        "id": "shipping-fee-below-500",
        "message": "Se eu comprar um ukulele de R$ 159,90, quanto custa o frete para Campo Grande?",
        "tools": {"policy_check"},
        "must": ["35"],
        "must_not": [],
    },
    {
        "id": "expired-return-window",
        "message": "Comprei um violão há 2 meses e me arrependi. Posso devolver?",
        "tools": {"policy_check"},
        "must": ["7"],
        "must_not": ["posso devolver", "sem problema"],
    },
    {
        "id": "defect-window-expired",
        "message": "O violão que comprei em dezembro veio com defeito. Ainda posso trocar?",
        "tools": {"policy_check"},
        "must": ["30"],
        "must_not": [],
    },
    {
        "id": "no-order-id-still-explains",
        "message": "Comprei um violão no ano passado e ele está com defeito. E agora?",
        "tools": {"policy_check"},
        "must": ["30"],
        "must_not": [],
    },
    {
        "id": "order-status",
        "message": "Como está o meu pedido 3?",
        "tools": {"get_order"},
        "must": [],
        "must_not": [],
    },
    {
        "id": "order-without-tracking",
        "message": "Qual o código de rastreio do pedido 20?",
        "tools": {"get_order"},
        "must": [],
        "must_not": ["BR"],
    },
    {
        "id": "expired-promotion",
        "message": "Vi uma promoção de 20% no violão Yamaha FG800, ainda vale?",
        "tools": {"get_active_promotions"},
        "must": [],
        "must_not": [],
    },
    {
        "id": "out-of-scope",
        "message": "Vocês vendem cordas de violão e palhetas?",
        "tools": set(),
        "must": ["instrumento"],
        "must_not": ["R$"],
    },
    {
        "id": "opening-hours",
        "message": "Que horas vocês fecham no sábado?",
        "tools": {"policy_check"},
        "must": ["13"],
        "must_not": [],
    },
    {
        "id": "policy-citation",
        "message": "Qual é a política de troca por preferência?",
        "tools": {"policy_search"},
        "must": ["§"],
        "must_not": [],
    },
]


def run_retrieval() -> int:
    index = PolicyIndex()
    print(f"mode: {'dense + lexical' if index.vectors else 'lexical only'}")
    top1 = top3 = 0
    for query, expected in RETRIEVAL_CASES.items():
        wanted = {expected} if isinstance(expected, str) else set(expected)
        hits = [hit["section"] for hit in index.search(query, k=3)]
        first = bool(wanted & set(hits[:1]))
        within = bool(wanted & set(hits))
        top1 += first
        top3 += within
        flag = "ok  " if first else ("top3" if within else "MISS")
        print(f"  {flag} {hits}  <- {query}")
    total = len(RETRIEVAL_CASES)
    print(f"\nretrieval: top-1 {top1}/{total} · top-3 {top3}/{total}")
    return 0 if top3 == total else 1


def run_agent() -> int:
    from app.agent import run_turn

    passed = 0
    for case in AGENT_CASES:
        result = run_turn(case["message"])
        reply = result["reply"]
        called = {call["name"] for call in result["tool_calls"]}
        missing_tools = case["tools"] - called
        missing_terms = [term for term in case["must"] if term.lower() not in reply.lower()]
        forbidden = [term for term in case["must_not"] if term.lower() in reply.lower()]
        ok = not missing_tools and not missing_terms and not forbidden
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case['id']}  tools={sorted(called)}")
        if not ok:
            print(f"        missing tools: {sorted(missing_tools)}")
            print(f"        missing terms: {missing_terms}")
            print(f"        forbidden:     {forbidden}")
            print(f"        reply: {reply[:300]}")
    print(f"\nagent golden set: {passed}/{len(AGENT_CASES)}")
    return 0 if passed == len(AGENT_CASES) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", action="store_true", help="retrieval metrics only")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    if args.json:
        from app.rag import PolicyIndex as _Index

        index = _Index()
        report = {
            query: [hit["section"] for hit in index.search(query, k=3)]
            for query in RETRIEVAL_CASES
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    return run_retrieval() if args.retrieval else run_agent()


if __name__ == "__main__":
    raise SystemExit(main())
