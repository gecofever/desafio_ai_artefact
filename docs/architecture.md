# Architecture

## Shape

Three layers, one direction of dependency. Nothing below ever imports anything
above it.

```
        ┌─────────────────────────────────────────────┐
        │ interface        app/ui/streamlit_app.py    │  chat, tool trail
        ├─────────────────────────────────────────────┤
        │ orchestration    app/agent.py               │  prompt + loop
        │                  app/tools.py               │  8 tool schemas
        ├─────────────────────────────────────────────┤
        │ knowledge        app/queries.py  (SQL)      │
        │                  app/policy.py   (numbers)  │
        │                  app/rag.py      (prose)    │
        ├─────────────────────────────────────────────┤
        │ storage          store.db  +  index/        │
        │                  app/db.py, app/text.py     │
        └─────────────────────────────────────────────┘
```

`app/sessions.py` and `app/config.py` are cross-cutting: configuration is read
once at import, persistence is written only by the interface.

## Modules and their contract

| Module | Owns | Must never |
|---|---|---|
| `app/config.py` | env loading, paths, model name, round cap | contain logic |
| `app/text.py` | `fold()` — lowercase, strip accents, collapse plurals | diverge from what the DB stores |
| `app/db.py` | CSV → SQLite, schema, folded search columns | transform a business rule |
| `app/queries.py` | SQL for catalogue, orders, customers, promotions | know about prompts or tools |
| `app/policy.py` | every threshold, window and percentage | read the clock implicitly |
| `app/rag.py` | manual → 28 sections → hybrid retrieval, citations | raise on a retrieval failure |
| `app/tools.py` | the model-facing surface: 8 tools returning JSON | compute a number |
| `app/agent.py` | persona prompt, tool loop, history replay | calculate or remember a fact |
| `app/sessions.py` | conversation persistence, Markdown export | store the tool trail for replay |
| `app/ui/` | rendering and `st.session_state` | contain business logic |

## A turn, in detail

1. `run_turn(message, history)` builds the prompt: system text (persona, rules,
   today's date in `America/Campo_Grande`), the replayed text history, the new
   message.
2. The model is called with all 8 tools bound. It either answers or asks for
   tools.
3. Each tool call is executed in process, its JSON result is appended as a
   `ToolMessage` with the matching `tool_call_id`, and the loop continues.
4. The loop stops when the model answers without a tool call, or after
   `MAX_TOOL_ROUNDS` (8). At the cap, one final call is made with tools removed
   and an instruction to answer with what is known — the agent degrades into a
   partial answer instead of spinning.
5. The interface renders the reply and the trail; both are persisted.

### Why the loop is hand-written

An executor hides exactly the part a reviewer needs to see: when does the model
get to speak without evidence, and what happens when a tool fails. Here a tool
failure becomes a JSON error the model can recover from
(`{"error": "tool_failed", ...}`), the round cap is explicit, and the whole
control flow fits on one screen.

### Why history is text-only

Stored turns are replayed as `HumanMessage` / `AIMessage` with text content only.
Two reasons, both learned the hard way:

- the vendor contract for an assistant turn with tools is `tool_calls` plus
  `role: "tool"` messages carrying `tool_call_id`; a stored
  `{name, arguments, result}` shape is rejected and the failure surfaces as a
  friendly apology on *every second turn*;
- even if the shape were accepted, replaying yesterday's stock level biases the
  answer. Each turn re-queries.

## Retrieval design

`read_pdf_lines()` uses pypdf's layout mode so headings keep their own line.
`split_sections()` then:

1. matches numbered heading lines (`^\d{1,2}(\.\d{1,2})?\.?\s+…`);
2. keeps only candidates that advance the numbering (`1, 1.1, 1.2, 2, …`) —
   this is what rejects `12.345.678/0001-90` and `09:00 às 18:00`;
3. attaches the body up to the next heading and records the page.

`PolicyIndex.search()` blends:

- **dense**: `text-embedding-3-small` vectors, cached in `index/embeddings.json`,
  cosine similarity, weight 0.65;
- **lexical**: BM25 with accent/plural folding, a pt-BR stopword list and a
  concept lexicon (customer vocabulary → manual vocabulary), weight 0.35.

If there are no vectors (no API key, first run, network failure) the lexical
score is used alone and the tool still returns citable sections.

## Failure behaviour

| Failure | What happens |
|---|---|
| No `OPENAI_API_KEY` | `run_turn` raises a single explicit error; the UI shows it in the chat |
| Embedding API down | retrieval falls back to BM25 silently, answers still cite sections |
| Database missing | sidebar tells the operator to run `scripts/build_db.py` |
| Tool raises | caught, returned as JSON `tool_failed`; the model explains it |
| Round cap hit | final answer without tools, flagged `truncated: true` |

## Cost and latency

A single-tool turn is 2 model calls and 1 retrieval embedding. Two-tool turns are
common (catalogue + policy). Measured on this machine: 2–5 s per turn with
`gpt-4.1`, cents per conversation. The index build is 28 embeddings, once.

## Extending

- **New data source**: add a query function in `app/queries.py` and a thin tool
  in `app/tools.py`. The agent changes only if the tool description changes.
- **New policy**: add the number to `app/policy.py`, a topic to
  `policy_lookup`, and a test asserting the boundary value. The prompt mentions
  the rule in one line at most; it never owns it.
- **New manual version**: drop the PDF in `data/`, rerun
  `python scripts/build_db.py --index`. `tests/test_rag.py` fails loudly if the
  section list changed, which is the intended alarm.
