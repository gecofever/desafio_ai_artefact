# Empório da Música — customer service agent

A support agent for a musical instrument store. It answers questions about the
catalogue, orders and store policies, and it answers them **from the store's own
data**:

> Prices, stock and order status come from SQLite.
> Rules with numbers — return windows, instalments, freight, discounts — come
> from a policy engine in code.
> Policy prose and its citation come from a retrieval layer over the manual.
> The model only decides what to look up and how to say it.

The whole point of the project is that boundary. A language model that is allowed
to remember a price or compute a deadline will eventually produce a confident
wrong answer, and in retail that is a legal problem, not a bug. So the model is
never given the chance: every fact it can state, it fetched through a tool on
that turn.

## What is in the box

- **8 tools** the model can call (catalogue, orders, customers, promotions, the
  deterministic policy engine, and retrieval over the manual).
- **A policy engine** (`app/policy.py`) where every number lives, with `today`
  injected so windows are testable.
- **A retrieval layer** (`app/rag.py`) that splits the 8-page manual into 28
  numbered sections and returns the section that answers a question, with its
  citation (`§4.1`).
- **A Streamlit interface** that keeps the chat clean by default; the tool
  trail is an audit tool, toggled from the sidebar ("Ver fontes consultadas —
  modo auditoria"), so the customer sees a normal conversation and the
  reviewer can turn the trace on when needed.
- **44 tests** (40 backend + 4 UI) and **an evaluation harness** with a golden set of 14 questions.

## Quick start

```bash
git clone git@github.com:gecofever/desafio_ai_artefact.git
cd desafio_ai_artefact

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill OPENAI_API_KEY
python scripts/build_db.py --index   # CSVs -> store.db, manual -> index/

pytest -q                   # 40 tests
streamlit run app/ui/streamlit_app.py
```

Open <http://localhost:8501>. Nothing else is required: no Docker, no build step,
no CDN. Python 3.12+ (developed and verified on 3.14).

Two optional commands measure the system:

```bash
python scripts/eval.py --retrieval   # retrieval accuracy, cheap, no agent
python scripts/eval.py               # golden set through the live agent
python scripts/run_transcripts.py    # re-record docs/conversations/
```

## How one turn works

```
customer message
      |
      v
ChatPromptTemplate (persona + rules + today's date)
      |
      v
ChatOpenAI.bind_tools(8 tools)  <-----------------------+
      |                                                 |
      +--> no tool call -----------------> answer       |
      |                                                 |
      +--> tool call -> execute -> ToolMessage ---------+
             |
             +-- search_products / get_order / ... -> SQLite
             +-- policy_check -------------------> app/policy.py (numbers)
             +-- policy_search ------------------> app/rag.py (prose + citation)
```

The loop is in `app/agent.py` and it is about 25 lines. History is replayed as
text only: tool calls and tool results from previous turns are deliberately
discarded, because a stock level from three messages ago is stale and because a
malformed vendor message shape is the classic way to break the second turn of a
conversation.

## The five decisions, and why

The brief says the choices are mine as long as the README justifies them. These
are the five anyone would ask about, with the trade-off I accepted.

### 1. CSVs into SQLite — yes

The data is six CSVs, 65 products, 20 orders. SQLite costs nothing to deploy, is
a single file the reviewer can open, and — the actual reason — it lets the
availability rule live **once**. `status = 'active' AND stock_quantity > 0`
appears in one place (`app/db.py` / `app/queries.py`), so the agent cannot answer
about a discontinued instrument using a different definition of "available" than
the search tool. A CSV scan in Python would have re-implemented that predicate
per tool.

Where it stops being the right answer: concurrent writes, or a second service
reading the same data. Then Postgres, and the tool layer does not change.

### 2. Policy rules in code — yes, and this is the non-negotiable one

Every number the agent can say out loud is computed in `app/policy.py`:
the 7-day regret window, the 30-day defect window, the 90-day legal warranty,
the R$ 500 free-shipping threshold, the R$ 35,00 fee, the PIX 5% (never
cumulative), the 1–3x / 4–6x / 7–12x instalment tiers, store hours.

Two details matter more than the list:

- **The window is anchored on delivery, not on the order date.** Counting 7 days
  from the order date can produce a deadline that closes before the parcel
  arrives — the customer is told their return window ended before the product
  shipped. `policy_check(order_id=...)` reads the order and anchors itself.
- **The engine returns a decision, not a formula.** It answers
  `{"eligible": false, "deadline": "2025-12-12", "days_left": 0}`. The model
  phrases that; it has no arithmetic to get wrong.

The prompt is then a summary of the policy, never the source of it. If the manual
and the prompt disagree, the engine wins by construction, because the engine is
what the tool returns.

### 3. Retrieval over the manual — yes, for citation, not for context budget

The manual is 8 pages and would fit in the prompt. So retrieval is not here to
save tokens; it earns its place for three other reasons:

1. **Citation.** Answers that depend on a policy name the section (`§4.1`). A
   section id only exists if the document was split into sections somewhere.
2. **Precision.** Injecting one section instead of ten pages stops unrelated
   rules (privacy, loyalty, warranty exclusions) from bleeding into an answer
   about freight.
3. **Change.** Policies change per section. A section-indexed corpus can be
   re-indexed and diffed per version.

Numbers still come from `policy.py`. Retrieval returns prose and a citation —
that split is what keeps a retrieval miss from turning into a wrong number.

**How it splits.** `pypdf` in layout mode, so headings land on their own line; a
heading is a numbered line, and sections must form an increasing sequence
(`1, 1.1, 1.2, 2, …`). That sequence check is what rejects `12.345.678/0001-90`
and `09:00 às 18:00` pretending to be headings. A test asserts the exact 28
section ids, so a new manual version fails loudly instead of silently.

**How it retrieves.** Hybrid: dense embeddings (`text-embedding-3-small`, cached
under `index/`) combined with BM25 over accent- and plural-folded tokens. The
lexical scorer is not decoration — it is the offline fallback when the embedding
API is unavailable, and it carries the whole answer in that case. A small
Portuguese lexicon bridges customer vocabulary to manual vocabulary
("posso devolver?" → *arrependimento*, "que horas abre?" → *horário de
funcionamento*), because that is where a naive lexical retriever fails.

Measured on the golden set: **top-1 13/13, top-3 13/13** (`scripts/eval.py
--retrieval`).

### 4. Streamlit — yes, with a known trade

Streamlit gets a real chat UI in ~150 lines: chat history, a visible tool trail
per answer (toggleable from the sidebar as "Ver fontes consultadas — modo
auditoria"), sample questions, transcript export, session persistence. For a
case that must be reviewed by someone who has ten minutes, that is the right
ratio.

The tool trail is an audit tool, not chat furniture: by default the customer
sees a clean conversation; turning the toggle on reveals what the agent looked
up under each answer. That split matches how real customer service works — the
customer does not want to see the CRM, the support lead does.

The trade I accepted: Streamlit re-runs the script on every interaction, so state
management is explicit (`st.session_state` + SQLite) rather than free, and there
is no HTTP API to point a different client at. A FastAPI + static front-end would
be the stronger answer for a "build a product" brief; for "prove this design
works", Streamlit removes a day of UI work that adds nothing to the argument.

### 5. LangChain — yes, used narrowly and on purpose

LangChain is used where it is a clear win:

- `ChatPromptTemplate` + `MessagesPlaceholder` for the persona prompt;
- `@tool` deriving each tool's JSON schema from its signature and docstring, so
  the tool contract and the prompt cannot drift;
- `ChatOpenAI.bind_tools` and the message types.

It is **not** used to hide the loop: there is no `AgentExecutor`, no LangGraph
graph, no chain soup. The loop is 25 readable lines, which is what lets me defend
this project line by line in an interview. Adding a framework layer that I would
have to explain instead of the code I wrote is a net loss at this size.

## Data traps found in the CSVs (and how the code answers them)

| Trap | What the data shows | What the agent does |
|---|---|---|
| Stored total ≠ sum of items | Orders 3 and 20: 3 450,00 vs 3 498,00 and 1 400,00 vs 1 488,00 | `total_brl` is the source of truth; nothing recomputes it. `test_data.py` locks it. |
| Availability ≠ existence | 1 active product with stock 0, 2 discontinued, 1 coming soon | One predicate: `active AND stock > 0`. The others are reported as unavailable for a *reason*. |
| Promotions | 25 rows, 4 active | Only active promotions are returned; an expired one is answered as expired, with today's price. |
| Missing tracking | 12 of 20 orders have no `tracking_code` | Answers "not shipped yet". Never fabricates a code — the manual documents the `BR…BR` format, which makes an invented code look real. |
| Empty categories | Categories 6, 7, 8 have no products | Answered as "not in the catalogue", not as an error. |
| All orders are old | Newest order 2026-03-22, today 2026-11-01 | Every 7/30-day window is expired. The correct answer is the closing date plus the manufacturer's warranty. |
| Accents and plurals | "violão" written as `violao`, `violões`, `violoes` | One `fold()` used by the SQL index, the queries and the retriever. |
| `specs` is a JSON string | — | Parsed at load time; kept as JSON in the row. |

## Verification

What was actually run, not what is claimed:

- **44 tests** green (`pytest -q`): 40 backend (data invariants, tool behaviour,
  policy arithmetic, manual sectioning, retrieval, history-replay regression)
  plus 4 UI tests using Streamlit's `AppTest` — asserting the app renders
  without exceptions, the tool trail is hidden by default, the audit toggle
  surfaces it on demand, and the caption text adapts to the mode.
- **Golden set 14/14** through the live model (`scripts/eval.py`).
- **Retrieval 13/13 top-1** (`scripts/eval.py --retrieval`).
- **6 recorded conversations** in `docs/conversations/`, generated by
  `scripts/run_transcripts.py` — every one from a real run against the model.

Three defects found while verifying, all fixed and covered by a test:

1. **Multi-word search failed.** `"violão Yamaha"` returned nothing, because the
   word *violão* lives in the category and not in the product name — the agent
   told a customer there were no Yamaha guitars while twelve were in stock.
   Fixed with token-by-token matching over folded columns.
2. **Retrieval missed paraphrases.** Lexical-only scoring sent "posso devolver?"
   to the promotions section. Fixed with BM25 + the concept lexicon, and now
   measured by the harness.
3. **The prompt let the model answer a policy question with a request for the
   order number only.** With no order id, the correct behaviour is to state the
   rule and ask for the number in the same message. The golden set caught it;
   the prompt and the topic aliases were the fix.

## Limitations

- **The pause between the model and the tools is the weak point.** If the store
  later lets the model answer without a tool call for a "simple" question, this
  design degrades. The rule has to stay absolute.
- **No authentication.** A real deployment needs customer identity before it
  serves order data; today an order number is enough, which is exactly the
  problem the case is not asking about.
- **Policy versioning is by document, not by date.** The agent applies today's
  policy to an order placed under an older one. The honest fix is a version per
  order date; that is a week of work, not an afternoon.
- **The golden set is 14 cases.** It catches regressions in the traps, not in
  everything. It should grow with every prompt change.
- **Embedding cost and latency.** The index is cached, so it is one API call at
  build time, but each retrieval still pays a query embedding (tens of ms).
- **Cursor/streaming is not implemented**; the answer appears when it is ready.

## How AI assistants were used

Assistants wrote most of the module bodies and tests against a written contract
(what each module must accept and return, and what it must never do). What was
mine: the architecture and the boundary between data, rules and language; the
tool set; the policy engine's anchoring decisions; the data traps; the evaluation
harness; and the end-to-end verification that found the three defects above.

Where the assistants were wrong, the tests say so — the multi-word search bug was
written by an assistant and caught by running the thing, not by reading it.

## Repository layout

```
app/
  config.py          environment and constants
  db.py              CSV -> SQLite, with folded search columns
  text.py            the single fold() used everywhere
  queries.py         SQL queries (no LLM knowledge)
  policy.py          deterministic policy engine (all numbers)
  rag.py             manual -> 28 sections -> hybrid retrieval
  tools.py           the 8 LangChain tools (the model-facing surface)
  agent.py           persona prompt + the tool-calling loop
  sessions.py        conversation persistence + export
  ui/streamlit_app.py
data/                the challenge CSVs + policy manual
docs/                decisions, architecture, recorded conversations
scripts/             build_db, eval, run_transcripts
tests/               40 tests
```
