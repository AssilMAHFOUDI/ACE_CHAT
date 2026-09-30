# ACE CHAT

**ACE CHAT** is an intelligent chat application built with **Streamlit** and powered by **Google Gemini**. It combines a conversational assistant, a **vector-based RAG** (Retrieval-Augmented Generation) document analysis mode, and a **ReAct agent loop** that can autonomously call external tools (web search, weather, calculator). Conversation history and document embeddings are persisted in **Supabase**.

The codebase is layered (`app.py` -> `services/` -> `modules/`), validated by an
automated test suite (173 tests, 100% coverage) and checked by **Ruff** in CI.

---

## Features

- **Classic Chat** - Real-time conversation with the Gemini model, with context memory.
- **Document Analysis (Vector RAG)** - Upload one or more `.txt` / `.pdf` files. Each document is split into overlapping chunks, embedded with Gemini, and stored in Supabase, so a session builds a real **knowledge base** holding several documents.
- **Knowledge Base per Session** - The sidebar lists every indexed document with its chunk count, lets you delete **one document at a time** (🗑️) without touching the others, and lets you aim a question at *all documents* or at *one specific document*.
- **Sources** - Each retrieved excerpt is labelled with its document (`[Extrait de <fichier>]`) and the file names actually used are shown under the answer (`📎 Sources : ...`).
- **ReAct Agent Loop** - The model autonomously decides, step by step, whether to call a tool or produce a final answer. Tool results are fed back into the model until it is ready to respond (bounded to 10 iterations).
- **Reflection** - Every question goes through a planner first: it either answers directly (when the answer is already in the question or in the provided context) or returns a 2-4 step plan. An answer that used tools is self-reviewed once before being shown, and a repeated tool call is served from the cache instead of hitting the network again.
- **Streamed Answers** - The answer is written token by token as the model produces it, instead of appearing only once the full text is ready.
- **Tool Calling (Function Calling)** - Gemini can automatically invoke:
  - **Web search** (DuckDuckGo) for news, scores, and recent information.
  - **Weather** (Open-Meteo) for the current weather in a city.
  - **Calculator** for evaluating mathematical expressions.
- **Live Agent Status** - Streamlit status boxes show each reasoning step and tool call in real time.
- **Persistence** - Conversation history and document chunks are saved to Supabase.
- **Reset** - Clears chat history and document chunks, then starts a brand-new session.
- **Multilingual** - The assistant replies in the language of the question.

---

## Architecture

```
ACE_CHAT/
  app.py                        # Streamlit entry point (pure UI: sidebar + chat)
  pyproject.toml                # Packaging, Ruff, pytest and coverage configuration
  requirements.txt              # Runtime dependencies
  requirements-dev.txt          # Development dependencies (tests, linter)
  README.md
  .streamlit/
    secrets.toml                # API keys and credentials (not versioned)
  .devcontainer/
    devcontainer.json           # GitHub Codespaces / Dev Container config
  .github/
    workflows/ci.yml            # CI: Ruff + pytest with a 95% coverage gate
  models/
    schemas.py                  # Pydantic models validated on every Supabase read
  modules/
    __init__.py
    config.py                   # centralised settings, logging setup, model names
    ai_engine.py                # Gemini client, planner, streamed ReAct loop, RAG prompt
    reflexion.py                # Reflection: plan, self-critique, observations (pure logic)
    database.py                 # Supabase connection, history CRUD, vector search
    document_processor.py       # Text extraction, chunking, vectorization
    tools.py                    # Tools exposed to Gemini (web, weather, calc)
  services/
    session.py                  # Session id, chat history (load / save / reset)
    memoire.py                  # Bounded context: sliding window + running summary
    base_connaissance.py        # Knowledge base: index, list and delete documents
    recherche.py                # Question -> embedding -> search -> RAG prompt
  tests/
    conftest.py                 # In-memory fakes (Supabase, Gemini) and coverage theme
    test_*.py                   # 173 unit and integration tests
    htmlcov/                    # Generated coverage report (not versioned)
```

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI only: sidebar (mode, knowledge base, search scope), chat flow, live status boxes, ingestion progress bar. Every business action is delegated to `services/`. |
| `services/session.py` | Session identifier, chat history (load / save / reset) and role validation. |
| `services/memoire.py` | Bounded conversation context: keeps the last `MEMORY_WINDOW_SIZE` messages verbatim and replaces older ones with an incrementally updated summary. Summaries are batched until `MEMORY_MIN_OVERFLOW` messages spill out, so a long conversation does not pay one call per turn. |
| `modules/reflexion.py` | Reflection helpers, pure logic with no Gemini client: planner verdict reading (`REPONSE` / `PLAN` protocol), bounded self-critique of the draft answer and observation formatting. |
| `services/base_connaissance.py` | Knowledge base: list indexed documents, index a document (extract -> chunk -> embed -> replace), delete a single document. |
| `services/recherche.py` | RAG search: embed the question, call the Supabase RPC, build the context prompt, extract the sources. |
| `models/schemas.py` | Pydantic v2 models (`ChatMessage`, `DocumentSummary`, `DocumentChunk`) validating what comes back from Supabase. |
| `modules/ai_engine.py` | Gemini client init, history conversion, planner call, ReAct agent loop with streamed replies, tool dispatch, embeddings (`get_embeddings`), RAG prompt generation. |
| `modules/database.py` | Cached Supabase connection, chat history CRUD, document listing (`list_session_documents`), semantic chunk search (`search_relevant_chunks`), cleanup of one document or of the whole session (`clear_document_chunks`). |
| `modules/document_processor.py` | Extract text (`.txt` / `.pdf`), split into overlapping chunks, embed and store each chunk. |
| `modules/config.py` | Centralised constants (chunking, batching, RAG thresholds, agent loop, planning and streaming flags, reflection limits, network timeouts, conversation memory window and summarisation threshold, Gemini models, embedding dimensions) and logging setup. |
| `modules/tools.py` | `recherche_web`, `meteo`, and `calculatrice` functions callable by Gemini. |

The dependency direction is one-way: `app.py` (presentation) calls `services/` (business
logic), which calls `modules/` (infrastructure). A service never imports Streamlit, so the
same logic could be reused by a CLI or an API without modification.

---

## How the Agent Works (ReAct Loop)

`get_ai_response()` in `modules/ai_engine.py` implements a **ReAct (Reason + Act)** loop:

1. **The planner goes first** (see *Reflection* below). Its session is created with a copy of the compressed history and with tools disabled; it answers either `REPONSE` - it already holds the answer, which is returned immediately, with no agent session and no tool call - or `PLAN` followed by 2 to 4 numbered steps.
2. The compressed history plus the latest user message (prefixed with `[Plan à suivre]` when a plan was returned) are sent to a fresh Gemini chat session (`client.chats.create`), so the context is rebuilt on every turn instead of relying on server-side state.
3. If the model returns `function_calls`, the requested tools are executed and their results are sent back to the model as function responses (`types.Part.from_function_response`).
4. An identical call (same tool, same arguments) is **never executed twice**: the cached result is returned with a hint to change approach, and a second repetition ends the loop with a forced synthesis, so a spinning agent costs no extra network request.
5. The loop repeats (up to `max_iterations = 10`) until the model returns a plain text answer, at which point the draft is optionally self-reviewed (see *Reflection* below).
6. The reply is **streamed** (`chat.send_message_stream`, `AGENT_FLUX_ACTIVE`): a `texte_callback` receives the text accumulated so far, and the interface writes it as it comes. A round that ends up calling a tool clears that zone first, so its accompanying sentence ("let me check") is never mistaken for the answer. The same path serves the planner's direct answers and the forced synthesis.
7. A `status_callback` reports each reasoning step and tool call to the Streamlit UI (`st.status`). Reasoning and answer live in two separate zones (`st.container()` then `st.empty()`), so the streamed answer never lands inside the reasoning trace.

Automatic function calling is explicitly disabled (`automatic_function_calling=disable=True`) so the application keeps control over tool execution, error handling, and UI updates. A system instruction injects the current date for time-aware reasoning.

```
question
   |
   v
planner (tools disabled, history copied)
   |
   +-- REPONSE --> return text --> END            (no agent session at all)
   |
   +-- PLAN --> current_message = "[Plan à suivre] ..."
                    |
                    v
        send_message_stream(current_message)
                    |
                    +-- no function_calls --> stream the text --> END
                    |
                    +-- function_calls --> execute tool(s)
                                           |
                                           +--> current_message = tool result(s)
                                                     |
                                                     +--> loop again
```

### Conversation Memory (bounded context)

The context sent to Gemini is kept under a hard limit by `services/memoire.py`,
*before* `format_history_for_gemini()` runs:

1. At least the last `MEMORY_WINDOW_SIZE` messages (6) are always kept
   verbatim: the view handed to the model is never shorter than that window.
2. When older messages fall out of that window, they are summarised into a short
   block (`MEMORY_SUMMARY_MAX_CHARS` characters) injected as a leading
   `[Résumé des échanges précédents]` turn.
3. A new summary is only paid for once `MEMORY_MIN_OVERFLOW` (4) more messages
   have spilled out since the last one: until then those older messages are kept
   verbatim, so the view is slightly longer than the window rather than paying a
   call for one or two new messages - never shorter.
4. The summary is **incremental**: it is recomputed only when new messages spill
   out of the window, and the previous summary is folded into that call, so a
   long conversation does not pay a full re-summarisation at every turn. The
   number of messages already covered is cached in `st.session_state`
   (`resume` / `resume_jusqua`) and cleared by the *Recommencer* button.
5. If the summarisation call fails, the recent window is sent alone and the
   cached index does not advance, so the same messages are summarised again on
   the next turn: nothing is silently dropped.

Displayed history and Supabase persistence stay **complete**: only the view
handed to the model is compressed.

### Reflection (plan, self-critique, stagnation)

`modules/reflexion.py` holds the reflection logic. The critique helpers never
talk to Gemini themselves (the model call is injected through `appeler_modele`),
so the module stays pure, offline-testable and reusable by any other interface;
reading the planner's verdict is a plain text protocol and needs no model at all.

1. **Systematic planning** (`AGENT_PLAN_ACTIVEE`, `PLAN_MAX_CHARS`,
   `PLAN_MARQUEUR_MAX_CHARS`). Every question goes through a planner session first
   - tools disabled, with a **copy** of the compressed history, so it can judge a
   follow-up ("and in English?") without its own exchange polluting the agent
   session. It answers on a two-marker protocol:
   - `REPONSE` followed by the answer: the answer is already entirely in the
     question or in the context handed over (greeting, mental calculation,
     reformulation, question already covered by the retrieved excerpts). The text
     is returned as-is, marker stripped, and **the ReAct loop never opens**: the
     question costs a single call.
   - `PLAN` followed by 2 to 4 numbered steps: an external source (web, weather)
     or several chained steps are needed. The plan is shown in the status box,
     prepended to the first agent message as `[Plan à suivre]` and recalled in the
     forced synthesis.
   Anything else is read as `PLAN`, which keeps the failure mode safe: a plan is
   never served as an answer, and an empty or unreadable planner reply simply
   leaves the agent deciding alone. The streaming router waits for the first line
   (or `PLAN_MARQUEUR_MAX_CHARS` characters) before routing anything, so a plan is
   never published in the answer zone, while a direct answer streams live.
2. **Bounded self-critique** (`AGENT_CRITIQUE_ACTIVEE`, `CRITIQUE_MAX_CHARS`,
   `CRITIQUE_BROUILLON_MAX_CHARS`). Once the agent holds a text answer *and* has
   collected observations, the draft is reviewed once against the question and
   those observations (the recall sent to the model is capped by
   `OBSERVATIONS_MAX_CHARS`, most recent observations first). The model answers
   `OK` or `INSUFFISANT: <reason>`; a refusal relaunches the agent with the
   reason (inside the same iteration
   budget). A plain chat answer that used no tool is never reviewed, and an
   unreadable verdict, an empty answer or a failing call all **accept the draft**:
   reflection can downgrade quality, never block a reply.
3. **Stagnation and tool cache** (`STAGNATION_MAX_APPELS_IDENTIQUES`). Tool calls
   are keyed by `(name, sorted arguments)`. A repeat is served from the cache
   with a warning instead of hitting the network again; beyond the threshold, the
   agent stops exploring and concludes from what it already has.

Cost per question: one call for the planner, plus one per agent round when it
returned a plan (and per tool round), plus one for the critique when tools were
used, plus one more only if that critique was refused - all bounded by
`max_iterations`. A question the planner answers directly costs a single call:
exactly what it cost before planning existed.



---

## How the Vector RAG Works

When a document is uploaded in **Document Analysis** mode:

1. **Extraction** - `extract_text_from_file()` reads the text from the `.txt` or `.pdf`.
2. **Chunking** - `split_text_into_chunks()` splits the text into ~1000-character chunks with a 200-character overlap to preserve context across boundaries.
3. **Embedding** - the chunks are converted into 3072-dimensional vectors by `get_embeddings()`, which calls the `gemini-embedding-2` model **by batches** (one request per 20 chunks) instead of one request per chunk.
4. **Storage** - the previous version of **this very document** (same `session_id` + same `file_name`) is deleted first, then the new chunks are inserted in batches. The other documents of the session are left untouched: several documents can coexist in one session, and replacing a document never destroys its neighbours.

When a question is asked:

1. The question is embedded with the same model.
2. `search_relevant_chunks()` calls the Supabase RPC **`match_document_chunks`** to retrieve the most similar chunks (cosine similarity, `match_threshold = 0.3`, `match_count = 4`). The search is always filtered by `session_id` and, depending on the **🔎 Chercher dans** selector, either by all documents (`file_name = None`) or by one chosen document. Targeting a single document guarantees the answer cannot mix two sources.
3. The retrieved chunks are combined into a context and injected into the prompt by `generate_rag_prompt()`.
4. The model answers **only** from the provided context, or states it doesn't know. Each excerpt is labelled `[Extrait de <file_name>]` so several documents can be told apart and cited.
5. The document names actually used are listed under the answer (`📎 Sources : ...`).

This is a genuine retrieval pipeline (embeddings + similarity search), not prompt stuffing.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** - Web interface
- **Google GenAI SDK** (`google-genai`) - Chat model `gemini-3.5-flash-lite`, embedding model `gemini-embedding-2`
- **Supabase** - PostgreSQL + `pgvector` for history and document chunks
- **Pydantic v2** - Validation of the data read from Supabase
- **pypdf** - PDF text extraction
- **ddgs** - DuckDuckGo web search
- **Open-Meteo API** - Weather data (no API key required)
- **Ruff** - Linting and formatting (development only)
- **pytest + pytest-cov** - Test suite and coverage (development only)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AssilMAHFOUDI/ACE_CHAT.git
cd ACE_CHAT
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the development dependencies (optional)

Only required to run the test suite and the linter:

```bash
pip install -r requirements-dev.txt
```

---

## Configuration

Create the `.streamlit/secrets.toml` file at the project root (this file is ignored by Git):

```toml
GEMINI_API_KEY = "your_gemini_api_key"

[supabase]
url = "https://your-project.supabase.co"
key = "your_supabase_key"
```

> **Never commit this file.** It is already listed in `.gitignore`.

### Getting the keys

- **Gemini key**: [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Supabase**: create a project on [supabase.com](https://supabase.com), then get the URL and key from *Project Settings - API*.

### Prepare the Supabase database

The app requires the **`pgvector`** extension, two tables, and one RPC function.

#### Chat history table

```sql
create table chat_history (
  id bigint generated by default as identity primary key,
  session_id text not null,
  role text not null,
  content text not null,
  created_at timestamptz default now()
);
```

#### Document chunks table (with vectors)

```sql
-- Enable the vector extension
create extension if not exists vector;

create table document_chunks (
  id bigint generated by default as identity primary key,
  session_id text not null,
  file_name text,
  content text not null,
  embedding vector(3072),      -- 3072-dim vectors from gemini-embedding-2
  created_at timestamptz default now()
);
```

#### Semantic search function (RPC)

`database.py` calls an RPC named `match_document_chunks`, passing the session **and the document name** (`p_file_name`). Example implementation (to paste in the Supabase SQL editor):

```sql
-- La fonction est identifiée par sa signature : pour ajouter p_file_name on la
-- supprime d'abord (PostgreSQL ne sait pas modifier une signature existante).
drop function if exists match_document_chunks(vector, float, int, text);

create or replace function match_document_chunks(
  query_embedding vector(3072),
  match_threshold float,
  match_count int,
  p_session_id text,
  p_file_name text default null
)
returns table (
  id bigint,
  file_name text,
  content text,
  similarity float
)
language sql stable
as $$
  select
    document_chunks.id,
    document_chunks.file_name,
    document_chunks.content,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  where document_chunks.session_id = p_session_id
    and (p_file_name is null or document_chunks.file_name = p_file_name)
    and 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
$$;
```

> **Required migration:** the `p_file_name` parameter must exist in the database. Until the block above is executed, the application detects it (`⚠️ Filtre par document indisponible côté base` in the logs), shows a warning in the interface, and filters the excerpts itself: it asks the database for a wider slice of the session (5 × `match_count`, at least 20 rows) and keeps only the chosen document. Answers stay usable, but the result is approximate (the top-N is computed before the application-side filter).
>
> Returning `file_name` feeds the `[Extrait de <fichier>]` labels, the `📎 Sources` line and that fallback filtering.
>
> If the five-parameter call still answers `PGRST202` right after running the block, the PostgREST schema cache is stale: reload it with `notify pgrst, 'reload schema';` (or *Project Settings → API → Reload* in the Supabase dashboard) and retry.

#### Verify the document filter (reads your data, writes nothing)

This test targets a session that holds **several** documents automatically — with a single-document session, both queries trivially return the same thing:

```sql
-- A) whole session : several file_name values expected
select file_name, count(*) as chunks
from match_document_chunks(
  (select embedding from document_chunks
    where session_id = (select session_id from document_chunks
                        group by session_id having count(distinct file_name) > 1
                        order by min(id) limit 1)),
  -1.0, 50,
  (select session_id from document_chunks
    where session_id = (select session_id from document_chunks
                        group by session_id having count(distinct file_name) > 1
                        order by min(id) limit 1)),
  null)
group by file_name order by file_name;

-- B) one document only : a single file_name expected (the one passed)
select file_name, count(*) as chunks
from match_document_chunks(
  (select embedding from document_chunks
    where session_id = (select session_id from document_chunks
                        group by session_id having count(distinct file_name) > 1
                        order by min(id) limit 1)),
  -1.0, 50,
  (select session_id from document_chunks
    where session_id = (select session_id from document_chunks
                        group by session_id having count(distinct file_name) > 1
                        order by min(id) limit 1)),
  (select file_name from document_chunks
    where session_id = (select session_id from document_chunks
                        group by session_id having count(distinct file_name) > 1
                        order by min(id) limit 1)
    order by file_name limit 1))
group by file_name order by file_name;
```

If the five-parameter call returns `PGRST202` here too, re-run the migration block above.

---

## Running the App

```bash
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

### Via GitHub Codespaces / Dev Container

The project includes a `.devcontainer` configuration. In a Codespace, the app starts automatically on port `8501` after dependencies are installed.

---

## Testing and Code Quality

The project ships with an automated test suite (**173 tests**) covering `modules/`,
`services/` and `models/`, at **100% line coverage**.

```bash
python -m pytest
```

- Tests run **offline**: Supabase and Gemini are replaced by in-memory fakes defined in
  `tests/conftest.py`, so the suite needs no API key and no network access.
- Coverage and the HTML report are configured in `pyproject.toml`; the report is written to
  `tests/htmlcov/index.html` (custom dark theme in `tests/_theme_htmlcov.css`).
- Linting and formatting use **Ruff** (88 columns):

```bash
ruff check .            # static analysis: pycodestyle, pyflakes, isort, bugbear, pyupgrade
ruff format .           # formatting
```

- **Continuous integration**: `.github/workflows/ci.yml` runs `ruff check .`,
  `ruff format --check .` and `python -m pytest --cov-fail-under=95` on every push to
  `main` or `feature/*` and on every pull request.

`app.py` (the Streamlit layer) is outside the coverage scope and is verified manually.

---

## Usage

1. **Choose a mode** in the sidebar:
   - **Chat Classique** - Chat freely with the assistant (agent + tools).
   - **Analyse de Document** - Upload a `.txt` or `.pdf` file. It is chunked, embedded and stored with a progress bar, then added to the session's knowledge base. Uploading another file **adds** a document: it no longer deletes the previous ones.
2. **Manage the knowledge base** (document mode, sidebar):
   - Every indexed document is listed with its number of chunks.
   - The 🗑️ button deletes **only** that document.
   - *Recommencer la discussion* wipes the whole base and the chat history, then starts a new session.
3. **Aim your question** with the **🔎 Chercher dans** selector: *📚 Tous les documents* (default) searches the whole base, or pick one document to restrict the search to it.
4. **Ask a question** using the input bar at the bottom of the screen - it is embedded and the most relevant chunks are retrieved before the answer, which is followed by a `📎 Sources : ...` line. Once the conversation is long enough to be compressed, a second caption (`🧠 Mémoire : n message(s) le plus ancien(s) résumé(s) pour cet appel.`) reports how many older messages were folded into the summary for that call.
5. **Watch the agent work** - status boxes show each reasoning step and tool call live.
6. **Reset** the conversation with the *Recommencer la discussion* button - this deletes chat history and document chunks and starts a new session.

### Example prompts

- *"What was the score of Real Madrid's last match?"* -> triggers web search.
- *"What's the weather in Paris?"* -> triggers the weather tool.
- *"What is (45 * 12) / 3?"* -> triggers the calculator.
- In document mode: *"Summarize this document"*, then *"and in English?"* - or, with several documents: *"Compare my CV and my cover letter"* (search on *📚 Tous les documents*).

---

## Security

- Secrets (`GEMINI_API_KEY`, Supabase credentials) are stored in `.streamlit/secrets.toml`, excluded from version control.
- The `calculatrice` tool evaluates expressions with **`simpleeval`**, a restricted evaluator: built-ins, imports and dunder attribute access are unavailable, so an injection attempt such as `__import__("os").system("...")` is rejected with an error instead of being executed.

---

## Possible Improvements

- Retry the network tools with an exponential backoff on transient failures (a per-request timeout is already in place).
- Handle scanned (image-only) PDFs, where text extraction returns nothing: OCR (page rendering + a vision model) would cover them.
- Cover the Streamlit layer (`app.py`) with integration tests: it is currently outside the coverage scope.
- Replace the two text protocols (planner `REPONSE` / `PLAN`, critique `OK` / `INSUFFISANT: <reason>`) with structured Pydantic schemas.

---

## Version History

| Tag | Highlights |
| --- | --- |
| `v8.3.0` | Planning ahead of every question: the planner shares the compressed history and either answers directly (no agent session, a single call) or returns a 2-4 step plan; replies are streamed token by token, and the text of a tool round is cleared from the answer zone, 173 tests at 100% coverage |
| `v8.2.0` | Agent reflection: optional initial plan for substantial questions, bounded one-shot self-critique of tool-backed answers, identical tool calls served from a cache with stagnation detection, batched memory summaries, 163 tests at 100% coverage |
| `v8.1.0` | Resilient agent and bounded memory: forced synthesis when the ReAct loop runs out of iterations, per-request timeouts on network tools, UTF-8 encoding guard, sliding-window conversation memory with an incremental summary, 115 tests at 100% coverage |
| `v8.0.1` | README brought in line with the code: layered architecture, 87 tests at 99% coverage, Ruff and CI |
| `v8.0.0` | Knowledge base with several documents per session: per-document cleanup, search scope selector, sources under the answer, SQL filter by document, single-read file handling, answers in the language of the question, batched ingestion, Pydantic schemas, `services/` layer, 87 tests and CI |
| `v7.0.1` | Calculator hardened with `simpleeval` |
| `v7.0.0` | Chunking and Gemini embeddings, vector search in Supabase |
| `v6.0.0` | Agent version: ReAct loop with function calling |
| `v5.0.0` | Function calling: web search, weather, calculator |
| `v4.0.0` | Cloud persistence with Supabase, refactor into `modules/` |
| `v3.1.0` | PDF support for document analysis |
| `v3.0.0` | Document analysis (RAG) with memory |
| `v2.0.0` | Conversation history |
| `v1.0.0` | First release (Amnesia version) |

---

## License

This project is provided for learning purposes. Add a license if you intend to distribute it.