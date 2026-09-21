# ACE CHAT

**ACE CHAT** is an intelligent chat application built with **Streamlit**, powered by **Groq** (via its OpenAI-compatible API), and backed by **Supabase**. It pairs a conversational assistant with a **ReAct agent loop** that autonomously decides when to call external tools (web search, weather, calculator) and reports every reasoning step live in the UI.

> ### ⚠️ Branch `test/alwaysdata`
>
> This branch is the **Alwaysdata deployment variant** of the project. Two deliberate changes separate it from `main`:
>
> 1. **The engine moved from Google Gemini to Groq** (`openai/gpt-oss-120b`, called through the OpenAI-compatible endpoint). The original Gemini engine is still in the repository (`modules/ai_engine.py`) but **`app.py` no longer imports it**.
> 2. **Document Analysis (vector RAG) is disabled.** No embedding model and no `fastembed` are installed by this branch's `requirements.txt`, so the deployment stays within the Alwaysdata disk quota. Documents are neither embedded nor retrieved; see *Document Analysis (RAG) - disabled on this branch* below for what was removed and how to turn it back on.

---

## Features

- **Classic Chat** - Real-time conversation with the Groq-hosted model, with context memory.
- **ReAct Agent Loop** - The model decides, step by step, whether to call a tool or produce a final answer. Tool results are fed back into the model until it is ready to respond (bounded to 10 iterations).
- **Tool Calling (Function Calling)** - The model can invoke three tools:
  - **Web search** (`recherche_web`, DuckDuckGo via `ddgs`) for news, scores, and recent information.
  - **Weather** (`meteo`, Open-Meteo) for the current weather in any city (no API key needed).
  - **Calculator** (`calculatrice`, `simpleeval`) for evaluating mathematical expressions.
- **Self-Correcting Tool Calls** - Groq rejects hallucinated tools with a `tool_use_failed` (HTTP 400) error. `get_ai_response()` catches the `BadRequestError`, injects a corrective system message listing the real tools, and retries instead of crashing the run.
- **Live Agent Status** - A Streamlit status box (`st.status`) displays each reasoning step and each tool call in real time.
- **Persistence** - Conversation history is stored in Supabase and reloaded on refresh; the *Recommencer la discussion* button clears it and starts a brand-new session.
- **Multilingual** - The assistant replies in the language of the question.
- **Startup Diagnostics** - `app.py` logs the public egress IP (country / city / org, via `ipinfo.io`) at boot so hosting issues can be diagnosed from the server logs.
- **Not available on this branch** - document upload and vector RAG (see below).

---

## Architecture

```
ACE_CHAT/
  app.py                        # Streamlit entry point (UI + orchestration)
  requirements.txt              # Python dependencies
  README.md
  .streamlit/
    secrets.toml                # API keys and credentials (not versioned)
  .devcontainer/
    devcontainer.json           # GitHub Codespaces / Dev Container config
  modules/
    __init__.py
    ai_engine_groq.py           # ACTIVE engine: Groq client, ReAct loop, tool schemas
    ai_engine.py                # legacy Gemini engine, kept for reference (not imported)
    database.py                 # Supabase connection, chat history CRUD (+ RAG helpers)
    document_processor.py       # Text extraction / chunking / storage (RAG only, not wired)
    tools.py                    # Tools exposed to the model (web, weather, calc)
```

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI, session management, orchestration, live status box, startup egress-IP log. |
| `modules/ai_engine_groq.py` | **Imported by `app.py`.** Groq/OpenAI client (`init_ai_client`), history conversion (`format_history_for_gemini`), system prompt (`_build_system_prompt`), OpenAI-style tool schemas (`TOOLS_SCHEMA`), ReAct loop (`get_ai_response`), tool dispatch (`_run_tool`), UI notifications (`_notify_tool_use`). |
| `modules/ai_engine.py` | **Legacy.** Identical public interface implemented with `google-genai` (chat model `gemini-3.5-flash-lite`, plus `get_embedding` and `generate_rag_prompt`). Swap the import in `app.py` to use it again. |
| `modules/database.py` | Cached Supabase connection (`init_connection`), chat history CRUD (`get_chat_history`, `save_message`, `clear_chat_history`), and RAG helpers kept for later (`search_relevant_chunks`, `clear_document_chunks`). |
| `modules/document_processor.py` | RAG ingestion: extract text from `.txt`/`.pdf` (`extract_text_from_file`), split into overlapping chunks (`split_text_into_chunks`), store them (`process_and_store_document`). On this branch it stores raw text only - no embeddings - and is **not called by `app.py`**. |
| `modules/tools.py` | `recherche_web`, `meteo` and `calculatrice` - the three functions callable by the model. |

---

## How the Agent Works (ReAct Loop)

`get_ai_response()` in `modules/ai_engine_groq.py` implements a **ReAct (Reason + Act)** loop:

1. The Streamlit history is converted into OpenAI-style `{"role", "content"}` messages by `format_history_for_gemini()` - the function keeps its original name so `app.py` stays compatible with either engine.
2. A `system` message built by `_build_system_prompt()` is prepended. It injects the current date, the list of allowed tools, and explicitly forbids inventing tools (`open_file`, `read_url`, `browse`, ...).
3. Each iteration calls:

   ```python
   client.chat.completions.create(
       model="openai/gpt-oss-120b",
       messages=messages,
       tools=TOOLS_SCHEMA,
       tool_choice="auto",
       temperature=0.3,
   )
   ```

4. If the answer contains `tool_calls`, each call is dispatched by `_run_tool()`; the results are appended as `{"role": "tool", "tool_call_id": ..., "content": ...}` and the loop continues.
5. If Groq raises `BadRequestError` (`tool_use_failed`, HTTP 400 - the model hallucinated a tool), the error is logged, a corrective `system` message is appended, and the loop retries.
6. As soon as the model answers with plain text, that text is returned to the UI. If the 10 iterations are exhausted, a fallback apology string is returned instead.
7. A `status_callback` (wired to `update_ui_status()` in `app.py`) reports every step to the Streamlit status box.

```
question
   |
   v
chat.completions.create(tools=TOOLS_SCHEMA, tool_choice="auto")
   |
   +-- tool_calls ----------------------> _run_tool(tool) --> role:"tool" --> loop
   |
   +-- BadRequestError(tool_use_failed) -> corrective system message -------> loop
   |
   +-- plain text ----------------------> return to UI -------------------> END
```

Unlike the legacy Gemini engine, which disabled SDK-side function calling (`automatic_function_calling=disable=True`), tool execution here is driven entirely by the application: the model only *requests* a tool, the app runs it, handles the errors, and updates the UI.

### The three tools

| Tool | Backing service | Notes |
| --- | --- | --- |
| `recherche_web(requete)` | DuckDuckGo (`ddgs`) | Returns the 3 best hits formatted as `Titre / Résumé / Lien`. |
| `meteo(ville)` | Open-Meteo geocoding + forecast | Resolves the city, then reads the current temperature, humidity and a French-labelled WMO weather code. |
| `calculatrice(expression)` | `simpleeval` | Safe evaluation; `InvalidExpression`, `ZeroDivisionError`, etc. are caught and returned as text so the model can read the failure. |

---

## Document Analysis (RAG) - disabled on this branch

On `test/alwaysdata` the document mode is intentionally switched off to keep the deployment small: the local embedding model pulled in by `fastembed` did not fit the Alwaysdata disk quota.

- `app.py` imports only `init_connection`, `get_chat_history`, `save_message` and `clear_chat_history` from `modules/database.py`; it never calls `modules/document_processor.py`.
- The sidebar shows an information banner instead of a file uploader: *"Mode « Analyse de Document » (RAG) désactivé pour ce test."*
- Incoming prompts are sent as-is (`prompt_pour_ia = prompt`), without any retrieval step.

### What still exists in the codebase

| Item | Where |
| --- | --- |
| Text extraction (`.txt` / `.pdf`), chunking, storage | `modules/document_processor.py` (`extract_text_from_file`, `split_text_into_chunks`, `process_and_store_document` - raw text only, no embedding) |
| Semantic search and chunk cleanup | `modules/database.py` (`search_relevant_chunks`, `clear_document_chunks`) |
| Embedding and RAG prompt building | `modules/ai_engine.py` (`get_embedding` with `gemini-embedding-2`, `generate_rag_prompt`) |
| `document_chunks` table and `match_document_chunks` RPC | Supabase - see the SQL below |

### How it worked on `main` (kept for reference)

**Ingestion:** extract the text -> split it into ~1000-character chunks with a 200-character overlap -> embed each chunk into a 768-dimensional vector -> store the chunk and its vector in `document_chunks`.

**Question answering:** embed the question with the same model -> `search_relevant_chunks()` calls the `match_document_chunks` RPC (cosine similarity, `match_threshold = 0.3`, `match_count = 4`, filtered by `session_id`) -> `generate_rag_prompt()` turns the retrieved chunks into a context block -> the model answers **only** from that context, or states that it does not know.

### To re-enable document mode on this branch

1. Add `pypdf` **and** an embedding provider to `requirements.txt` (either `google-genai` for the Gemini embedding model, or a local model such as `fastembed` if the disk quota allows it).
2. Implement the embedding call inside `process_and_store_document()` and insert the resulting vector into the `embedding` column.
3. Restore the uploader plus `get_embedding` at ingestion time, and the retrieval / `generate_rag_prompt` block in `app.py`.
4. Make sure `document_chunks.embedding vector(768)` and the `match_document_chunks` function exist in Supabase (SQL below).

> **Note:** `modules/document_processor.py` does `import pypdf` at the top of the file, but `pypdf` is **not** listed in `requirements.txt` on this branch (it is unnecessary while the RAG mode is off). Add it back before re-enabling ingestion.

---

## Tech Stack

- **Python 3.11+** (the devcontainer image is 3.11)
- **Streamlit** `1.63.0` - web interface, session state, status boxes
- **Groq API via the `openai` SDK** - chat model `openai/gpt-oss-120b` at `https://api.groq.com/openai/v1`, 30 s timeout
- **Supabase** - PostgreSQL storage for the chat history (plus `pgvector` support if RAG is re-enabled)
- **`ddgs`** - DuckDuckGo web search
- **`simpleeval`** - safe evaluation of mathematical expressions
- **Open-Meteo API** - geocoding + weather data (no API key required)
- **`ipinfo.io`** - egress IP lookup at startup
- **`pypdf`** - PDF text extraction *(RAG mode only, see the note above)*
- **`google-genai`** - legacy Gemini engine only (`modules/ai_engine.py`)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AssilMAHFOUDI/ACE_CHAT.git
cd ACE_CHAT
git checkout test/alwaysdata
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

`requirements.txt` on this branch:

```
streamlit==1.63.0
supabase>=2.3.0
ddgs==9.16.0
simpleeval==0.9.13
openai>=1.40.0
```

---

## Configuration

Create the `.streamlit/secrets.toml` file at the project root (this file is ignored by Git):

```toml
GROQ_API_KEY = "your_groq_api_key"

[supabase]
url = "https://your-project.supabase.co"
key = "your_supabase_key"
```

Only `GROQ_API_KEY` is required on this branch. `GEMINI_API_KEY` is used solely by the legacy `modules/ai_engine.py` (and by the embedding / RAG helpers), so it is optional here.

> **Never commit this file.** It is already listed in `.gitignore`.

### Getting the keys

- **Groq key**: [console.groq.com/keys](https://console.groq.com/keys)
- **Gemini key** *(only for the legacy engine or a RAG revival)*: [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Supabase**: create a project on [supabase.com](https://supabase.com), then copy the URL and key from *Project Settings - API*.

---

## Prepare the Supabase database

### Chat history table (required)

```sql
create table chat_history (
  id bigint generated by default as identity primary key,
  session_id text not null,
  role text not null,
  content text not null,
  created_at timestamptz default now()
);
```

### Document chunks table (RAG only - optional on this branch)

```sql
-- Enable the vector extension
create extension if not exists vector;

create table document_chunks (
  id bigint generated by default as identity primary key,
  session_id text not null,
  file_name text,
  content text not null,
  embedding vector(768),       -- 768-dim vectors from gemini-embedding-2
  created_at timestamptz default now()
);
```

### Semantic search function (RPC - RAG only)

`database.py` calls an RPC named `match_document_chunks`. Example implementation:

```sql
create or replace function match_document_chunks(
  query_embedding vector(768),
  match_threshold float,
  match_count int,
  p_session_id text
)
returns table (
  id bigint,
  content text,
  similarity float
)
language sql stable
as $$
  select
    document_chunks.id,
    document_chunks.content,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  where document_chunks.session_id = p_session_id
    and 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
$$;
```

---

## Running the App

Always launch the app **from the repository root** - the imports are absolute (`from modules...`):

```bash
streamlit run app.py
```

The app is then available at [http://localhost:8501](http://localhost:8501).

### Via GitHub Codespaces / Dev Container

The repository ships a `.devcontainer` configuration: it installs `requirements.txt` and starts `streamlit run app.py` automatically on port `8501` (with CORS and XSRF protection disabled for the forwarded preview).

### Deploying on Alwaysdata (the point of this branch)

- Install `requirements.txt` inside a virtual environment on the account and run `streamlit run app.py` from your site configuration, listening on the port Alwaysdata assigns.
- Keep `requirements.txt` free of heavy binary dependencies - `fastembed` and the local embedding model are deliberately absent so the account stays within its disk quota.
- On startup the logs contain a line such as:
  `🌍 [EGRESS IP] ip=... country=... city=... org=...`
  It comes from `log_egress_ip()` in `app.py` and shows which IP/region the host actually egresses from (handy when a provider blocks traffic from a datacenter country).
- Make sure `.streamlit/secrets.toml` exists on the server with `GROQ_API_KEY` and the `[supabase]` credentials. `st.secrets[...]` is read at import time, so a missing key stops the app immediately with a `KeyError`.

---

## Usage

1. Type your question in the input bar at the bottom of the page (*"Pose-moi une question..."*). The sidebar reminds you that document analysis is disabled on this branch.
2. Watch the agent work in the status box: each reasoning step (`Analyse et reflexion (Etape n)...`), each tool call (e.g. `Recherche sur le web : ...`), any invalid-tool correction, and finally `Redaction de la reponse finale...`.
3. The answer is rendered in the chat and persisted to Supabase together with your question, so the conversation survives a page refresh.
4. Use **🗑️ Recommencer la discussion** in the sidebar to erase the stored history and start a brand-new session id.

### Example prompts

- *"What was the score of Real Madrid's last match?"* -> triggers `recherche_web`.
- *"What's the weather in Paris?"* / *"Quelle est la météo à Dakar ?"* -> triggers `meteo`.
- *"What is (45 * 12) / 3?"* -> triggers `calculatrice`.

---

## Version History

The repository keeps one tag per learning milestone (see `git tag`):

| Tag | Milestone |
| --- | --- |
| `v1.0.0` | `amnesia` branch - first deployed version, no memory |
| `v2.0.0` | conversation history kept in the session |
| `v3.0.0` / `v3.1.0` | basic RAG with contextual memory, then PDF support |
| `v4.0.0` / `v4.0.1` | Supabase persistence, then refactor into `modules/` |
| `v5.0.0` | calculator and weather tools (function calling) |
| `v6.0.0` | web-search tool and the ReAct loop with live status UI |
| `v7.0.0` / `v7.0.1` | document embeddings stored in Supabase, calculator update |

`test/alwaysdata` is untagged: it branches off `main` right after `v7.0.1` and adds the Gemini -> Groq migration plus the RAG shutdown.

---

## Security

- Secrets (`GROQ_API_KEY`, `GEMINI_API_KEY`, the Supabase URL and key) live in `.streamlit/secrets.toml`, which is excluded from version control. Keep it that way and never paste real keys into source files, commits or issues.
- An untracked `config.yaml` may also hold a Gemini key and a database password. It is covered by `.gitignore`, but treat it as sensitive scratch space: delete it (or move its values into `secrets.toml`) and rotate any key that ever sat there.
- `calculatrice` uses `simpleeval` instead of `eval`, so an injection attempt such as `__import__("os").system("...")` is rejected as an invalid expression instead of being executed.
- `log_egress_ip()` performs an outbound request to `ipinfo.io` at every startup and writes the host's public IP, country, city and org to the application logs. Drop the call if that is not acceptable in your environment.
- There is no authentication: anyone who can reach the app URL uses the same Supabase tables, separated only by a random `session_id` generated per browser session.

---

## Possible Improvements

- Add `pypdf` (and an embedding provider) back to `requirements.txt` only when document mode is re-enabled.
- Batch the embeddings and the Supabase inserts to speed up ingestion.
- Add HTTP timeouts to the `urllib.request.urlopen` calls in `tools.py` (`meteo`).
- Bound/trim the conversation history sent to the model - it currently grows without limit.
- Cap invalid-tool retries so a repeatedly failing call cannot eat the whole 10-iteration budget.
- Add automated tests for the pure functions (`format_history_for_gemini`, `split_text_into_chunks`, `calculatrice`, tool registry vs. tool schema consistency).
- Introduce explicit planning and self-critique to strengthen the agent's autonomy.

---

## License

This project is provided for learning purposes. Add a license if you intend to distribute it.

