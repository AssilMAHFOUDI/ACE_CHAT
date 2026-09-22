# ACE CHAT

**ACE CHAT** is an intelligent chat application built with **Streamlit** and powered by **Google Gemini**. It combines a conversational assistant, a **vector-based RAG** (Retrieval-Augmented Generation) document analysis mode, and a **ReAct agent loop** that can autonomously call external tools (web search, weather, calculator). Conversation history and document embeddings are persisted in **Supabase**.

---

## Features

- **Classic Chat** - Real-time conversation with the Gemini model, with context memory.
- **Document Analysis (Vector RAG)** - Upload one or more `.txt` / `.pdf` files. Each document is split into overlapping chunks, embedded with Gemini, and stored in Supabase, so a session builds a real **knowledge base** holding several documents.
- **Knowledge Base per Session** - The sidebar lists every indexed document with its chunk count, lets you delete **one document at a time** (🗑️) without touching the others, and lets you aim a question at *all documents* or at *one specific document*.
- **Sources** - Each retrieved excerpt is labelled with its document (`[Extrait de <fichier>]`) and the file names actually used are shown under the answer (`📎 Sources : ...`).
- **ReAct Agent Loop** - The model autonomously decides, step by step, whether to call a tool or produce a final answer. Tool results are fed back into the model until it is ready to respond (bounded to 10 iterations).
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
  app.py                        # Streamlit entry point (UI + orchestration)
  requirements.txt              # Python dependencies
  README.md
  .streamlit/
    secrets.toml                # API keys and credentials (not versioned)
  .devcontainer/
    devcontainer.json           # GitHub Codespaces / Dev Container config
  modules/
    __init__.py
    ai_engine.py                # Gemini client, ReAct loop, embeddings, RAG prompt
    database.py                 # Supabase connection, history CRUD, vector search
    document_processor.py       # Text extraction, chunking, vectorization
    tools.py                    # Tools exposed to Gemini (web, weather, calc)
```

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI, session management, orchestration, document ingestion trigger, live status boxes. |
| `modules/ai_engine.py` | Gemini client init, history conversion, ReAct agent loop, tool dispatch, embeddings (`get_embedding`), RAG prompt generation. |
| `modules/database.py` | Cached Supabase connection, chat history CRUD, document listing (`list_session_documents`), semantic chunk search (`search_relevant_chunks`), cleanup of one document or of the whole session (`clear_document_chunks`). |
| `modules/document_processor.py` | Extract text (`.txt` / `.pdf`), split into overlapping chunks, embed and store each chunk. |
| `modules/tools.py` | `recherche_web`, `meteo`, and `calculatrice` functions callable by Gemini. |

---

## How the Agent Works (ReAct Loop)

`get_ai_response()` in `modules/ai_engine.py` implements a **ReAct (Reason + Act)** loop:

1. The latest user message is sent to a Gemini chat session (`client.chats.create`), which keeps state across turns.
2. If the model returns `function_calls`, the requested tools are executed and their results are sent back to the model as function responses (`types.Part.from_function_response`).
3. The loop repeats (up to `max_iterations = 10`) until the model returns a plain text answer.
4. A `status_callback` reports each reasoning step and tool call to the Streamlit UI (`st.status`).

Automatic function calling is explicitly disabled (`automatic_function_calling=disable=True`) so the application keeps control over tool execution, error handling, and UI updates. A system instruction injects the current date for time-aware reasoning.

```
question
   |
   v
send_message(current_message)
   |
   +-- no function_calls --> return text --> END
   |
   +-- function_calls --> execute tool(s)
                          |
                          +--> current_message = tool result(s)
                                    |
                                    +--> loop again
```

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

- **Python 3.11+**
- **Streamlit** - Web interface
- **Google GenAI SDK** (`google-genai`) - Chat model `gemini-3.5-flash-lite`, embedding model `gemini-embedding-2`
- **Supabase** - PostgreSQL + `pgvector` for history and document chunks
- **pypdf** - PDF text extraction
- **ddgs** - DuckDuckGo web search
- **Open-Meteo API** - Weather data (no API key required)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AssilMAHFOUDI/ACE_CHAT.git
cd ACE_CHAT
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
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

## Usage

1. **Choose a mode** in the sidebar:
   - **Chat Classique** - Chat freely with the assistant (agent + tools).
   - **Analyse de Document** - Upload a `.txt` or `.pdf` file. It is chunked, embedded and stored with a progress bar, then added to the session's knowledge base. Uploading another file **adds** a document: it no longer deletes the previous ones.
2. **Manage the knowledge base** (document mode, sidebar):
   - Every indexed document is listed with its number of chunks.
   - The 🗑️ button deletes **only** that document.
   - *Recommencer la discussion* wipes the whole base and the chat history, then starts a new session.
3. **Aim your question** with the **🔎 Chercher dans** selector: *📚 Tous les documents* (default) searches the whole base, or pick one document to restrict the search to it.
4. **Ask a question** using the input bar at the bottom of the screen - it is embedded and the most relevant chunks are retrieved before the answer, which is followed by a `📎 Sources : ...` line.
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
- The `calculatrice` function evaluates expressions via `eval` with `__builtins__` disabled. **This is not a fully safe sandbox** - for production, consider a dedicated expression library such as `simpleeval` or `numexpr`.

---

## Possible Improvements

- Secure the calculator with a dedicated expression evaluator.
- Re-index a document without deleting it first (today: remove it with 🗑️, then upload it again).
- Add HTTP timeouts to network tools (`tools.py`).
- Bound/trim the conversation history sent to Gemini.
- Introduce explicit planning and self-critique to strengthen the agent's autonomy.

---

## License

This project is provided for learning purposes. Add a license if you intend to distribute it.