# Technical decisions

One page per decision, in the order the questions tend to arrive. Each entry
states the context, the choice, the consequence I accept, and the alternative I
rejected — because the alternative is what the reader actually wants to know.

---

## ADR-001 — SQLite is the system of record

**Context.** Six CSVs, 65 products, 50 customers, 20 orders. The agent needs
catalogue, order and promotion facts on every turn.

**Decision.** Load the CSVs into `store.db` at build time and query it from
`app/queries.py`.

**Why.** The decisive reason is not speed, it is *singularity*: the availability
predicate (`status = 'active' AND stock_quantity > 0`) exists once and every tool
uses it. With CSV scans per tool, the search tool and the product tool can drift
and disagree about whether something is for sale. SQLite also gives the reviewer
a single inspectable file, requires no server, and makes joins (order → items →
product → promotion) a one-liner instead of nested loops.

**Consequence accepted.** Writes serialise; the file must be rebuilt when data
changes; no concurrent access from a second process. For this workload that is
free.

**Rejected.** (a) Reading CSVs in memory per request: repeated parsing, and the
availability rule would be duplicated per tool. (b) Postgres: a server to
provision for 20 rows. (c) A vector database for the catalogue: semantic search
over product names is the wrong tool — the catalogue is small and structured.

**When this changes.** Concurrent writers, or a second service reading the same
data. The tool layer would not change.

---

## ADR-002 — Numbers live in code, never in the prompt

**Context.** The manual defines 7-day and 30-day windows, a 90-day legal
warranty, a R$ 500 free-shipping threshold, a R$ 35,00 fee, a non-cumulative 5%
PIX discount, and three instalment tiers.

**Decision.** `app/policy.py` computes all of it and returns decisions
(`{"eligible": false, "deadline": "2025-12-12", "days_left": 0}`). The prompt
describes the rules in prose but owns none of them.

**Why.** A model that does date arithmetic is a model that will one day tell a
customer their return window closed on a date that never existed. It also has no
stable notion of "today". Putting the arithmetic in a pure function with an
injected clock makes it unit-testable and makes the failure mode a `False`
instead of an invention.

**Consequence accepted.** Two places describe the same rule (engine and prompt),
so they can drift. Mitigation: the engine is always the value returned by the
tool, and the tests assert the boundary values, so drift shows up as a test
failure rather than a wrong answer.

**Anchoring detail.** Return windows count from **delivery**, not from the order
date. Using the order date can produce a deadline earlier than the delivery
itself. `policy_check(order_id=…)` reads the order and anchors itself, so the
model never passes a date.

**Rejected.** (a) Rules in the system prompt: unauditable and untestable.
(b) "Just ask the model to be careful": not a control.

---

## ADR-003 — Retrieval over the manual, for citation, not for context size

**Context.** The policy manual is 8 pages (28 numbered sections, ~9 400
characters). It fits in a prompt.

**Decision.** Split the manual into sections, index them, and retrieve the
relevant ones per question. Use the retrieved section for wording and require a
citation (`§4.1`).

**Why.** Retrieval here buys three things the context window does not:
**citation** (a section id has to exist to be cited), **isolation** (one section
instead of ten pages removes cross-talk — privacy rules do not leak into a
freight answer) and **change management** (sections can be re-indexed and diffed
per manual version).

**Consequence accepted.** A new failure mode: a retrieval miss. Two mitigations:
numbers never depend on retrieval (they come from the engine), and the retrieval
quality is measured by `scripts/eval.py --retrieval` rather than assumed —
currently 13/13 top-1 on the golden set.

**Rejected.** (a) Whole manual in the prompt: cheapest, but no citations and
every answer carries unrelated policy. (b) Chunk by fixed token window: destroys
section boundaries, which are the citation unit. (c) A vector-only index: fails
on exact terms ("§5.3", "PIX") and has no offline mode.

**When this changes.** If the corpus grows past a few dozen pages, keep the
structure (sections) and swap BM25 for a proper index.

---

## ADR-004 — Streamlit for the interface

**Context.** The case asks for something a reviewer can use, and the role is
full-stack.

**Decision.** A single Streamlit page: chat, tool trail per answer (toggled from
the sidebar as "Ver fontes consultadas — modo auditoria"), sample questions,
transcript export, session persistence.

**Why.** It delivers a usable, demonstrable UI in ~150 lines. The tool trail is
the important part — showing *why* the answer is what it is turns the agent from
a black box into something reviewable, and that is the argument this project is
making.

**Consequence accepted.** The trail is hidden by default (a clean, natural
conversation for the customer) and revealed on demand from the sidebar
(auditing). Streamlit
re-runs the script per interaction, so state must be explicit
(`st.session_state` + SQLite), and there is no HTTP API for another client. No
streaming, so the answer appears when it is complete.

**Rejected.** (a) FastAPI + static front-end: stronger for a product brief,
costs a day of UI work that adds no evidence to *this* argument. (b) CLI only:
no tool trail, no reviewer path. (c) Gradio: comparable, but less control over
layout.

---

## ADR-005 — LangChain, used narrowly

**Context.** The brief asks for the simplest solution that works; the role
expects familiarity with the ecosystem.

**Decision.** Use LangChain for the pieces that are unambiguous wins:
`ChatPromptTemplate`, `@tool` (schema generation from signature + docstring),
`ChatOpenAI.bind_tools`, and the message types. Write the loop myself.

**Why.** Tool schemas derived from the function signature cannot drift from the
function; prompt templating keeps the persona in one place; the message types
are the vendor contract already. An `AgentExecutor` or a LangGraph graph would
add a layer that has to be explained *instead of* the code, and this project is
small enough that the layer is pure cost.

**Consequence accepted.** Less "batteries included": no built-in memory
management, no callback-based tracing UI, no retries for free. All of them are
optional here.

**Rejected.** (a) No framework at all: viable, but hand-maintaining JSON schemas
for 8 tools is exactly the kind of duplication that gets stale. (b) Full
LangGraph agent: a debugging surface with no payoff at this size.

---

## ADR-006 — History is replayed as text only

**Context.** The interface needs conversation continuity.

**Decision.** Persist every turn, but replay only `role` and text to the model.
Tool calls and tool results are stored for the UI and never sent back.

**Why.** Two independent reasons. First, the vendor's contract for an assistant
turn that called tools is a specific shape (`tool_calls` array plus `role: "tool"`
messages with `tool_call_id`); a stored simplified shape is rejected with a 400
and — because agent loops catch exceptions — the failure surfaces as a friendly
apology on every *second* turn, which looks like a personality problem, not a
bug. Second, replaying a stale result silently feeds old facts forward.

**Consequence accepted.** The model does not remember what it looked up; it
looks it up again. That is one extra tool call and the correct behaviour.

---

## ADR-007 — One `fold()` for text

**Context.** Brazilian customers write `violao`, `violão`, `violões`, `violoes`
for the same word, and the CSV stores only one spelling. The manual uses formal
vocabulary ("direito de arrependimento") where customers use colloquial
("posso devolver?").

**Decision.** A single `app/text.py::fold()` used by the database loader, the SQL
queries and the retriever; plus a small concept lexicon in the retriever for
vocabulary bridging.

**Why.** Three normalisers would eventually disagree, and the symptom is an
agent that says a product does not exist while twelve are in stock — which is
exactly the bug found during verification (see README, "Verification").

**Consequence accepted.** The lexicon is hand-written and Portuguese-specific;
it is data, not logic, and it is covered by tests.

---

## ADR-008 — `total_brl` is the source of truth

**Context.** Orders 3 and 20 store a total that disagrees with the sum of their
items (3 450,00 vs 3 498,00; 1 400,00 vs 1 488,00).

**Decision.** Never recompute a total from items. Report `total_brl` and document
the discrepancy.

**Why.** `total_brl` is what the customer actually paid — it may include a
discount or a freight line the item table does not carry. Recomputing would make
the agent contradict the invoice, which is worse than an inconsistency in the
source data.

**Consequence accepted.** Any future reconciliation report must flag these two
orders explicitly. `tests/test_data.py` asserts exactly this pair, so a change in
the data cannot pass silently.
