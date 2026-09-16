# 🤖 ACE CHAT

**ACE CHAT** is an intelligent chat application built with **Streamlit** and powered by **Google Gemini**. It combines a conversational assistant, document analysis (RAG-style), and a **ReAct agent loop** that can autonomously call external tools (web search, weather, calculator), with conversation history persisted in **Supabase**.

---

## ✨ Features

- 💬 **Classic Chat** — Real-time conversation with the Gemini model, with context memory.
- 📄 **Document Analysis (RAG-style)** — Upload a `.txt` or `.pdf` file and ask questions about its content. The assistant answers using the document and the conversation history.
- 🧠 **ReAct Agent Loop** — The model autonomously decides, step by step, whether to call a tool or produce a final answer. Tool results are fed back into the model until it is ready to respond (bounded to 10 iterations).
- 🛠️ **Tool Calling (Function Calling)** — Gemini can automatically invoke:
  - 🔍 **Web search** (DuckDuckGo) for news, scores, and recent information.
  - 🌦️ **Weather** (Open-Meteo) for the current weather in a city.
  - 🧮 **Calculator** for evaluating mathematical expressions.
- 📡 **Live Agent Status** — A Streamlit status box shows each reasoning step and tool call in real time.
- 💾 **Persistence** — Conversation history is saved to Supabase and reloaded on each session.
- 🗑️ **Reset** — Button to clear the current conversation history.
- 🌍 **Multilingual** — The assistant replies in the language of the question.

---

## 🏗️ Architecture

```
ACE_CHAT/
├── app.py                        # Streamlit entry point (UI + orchestration)
├── requirements.txt              # Python dependencies
├── README.md
├── .streamlit/
│   └── secrets.toml              # API keys and credentials (not versioned)
├── .devcontainer/
│   └── devcontainer.json         # GitHub Codespaces / Dev Container config
└── modules/
    ├── __init__.py
    ├── ai_engine.py              # Gemini client, ReAct loop, RAG prompt
    ├── database.py               # Supabase connection + history CRUD
    ├── document_processor.py     # Text extraction (TXT / PDF)
    └── tools.py                  # Tools exposed to Gemini (web, weather, calc)
```

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI, session management, orchestration of calls, live status box. |
| `modules/ai_engine.py` | Gemini client initialization, history conversion, ReAct agent loop, tool dispatch, RAG prompt generation. |
| `modules/database.py` | Cached Supabase connection, read / write / delete messages. |
| `modules/document_processor.py` | Extract text from an uploaded file (`.txt` or `.pdf`). |
| `modules/tools.py` | `recherche_web`, `meteo`, and `calculatrice` functions callable by Gemini. |

---

## 🧠 How the Agent Works

`get_ai_response()` in `modules/ai_engine.py` implements a **ReAct (Reason + Act)** loop:

1. The latest user message is sent to a Gemini chat session.
2. If the model returns `function_calls`, the corresponding tools are executed and their results are sent back to the model as function responses.
3. The loop repeats (up to `max_iterations = 10`) until the model returns a plain text answer.
4. A `status_callback` reports each step to the Streamlit UI (`st.status`).

Automatic function calling is explicitly disabled (`automatic_function_calling=disable=True`) so the application controls tool execution and error handling. A system instruction injects the current date for time-aware reasoning.

> **Note:** This is a tool-using ReAct agent. It does not perform explicit planning, long-term memory, or self-critique. The "RAG" mode uses prompt stuffing (the full document is inserted into the prompt) rather than vector embeddings.

---

## 🧰 Tech Stack

- **Python 3.11+**
- **Streamlit** — Web interface
- **Google GenAI SDK** (`google-genai`) — Model `gemini-3.5-flash-lite`
- **Supabase** — PostgreSQL database for history
- **pypdf** — PDF text extraction
- **ddgs** — DuckDuckGo web search
- **Open-Meteo API** — Weather data (no API key required)

---

## 🚀 Installation

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

## 🔑 Configuration

Create the `.streamlit/secrets.toml` file at the project root (this file is ignored by Git):

```toml
GEMINI_API_KEY = "your_gemini_api_key"

[supabase]
url = "https://your-project.supabase.co"
key = "your_supabase_key"
```

> ⚠️ **Never commit this file.** It is already listed in `.gitignore`.

### Getting the keys

- **Gemini key**: [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Supabase**: create a project on [supabase.com](https://supabase.com), then get the URL and key from *Project Settings → API*.

### Prepare the Supabase database

Create a `chat_history` table with the following structure:

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `int8` / `bigint` | Primary key, auto-incremented |
| `session_id` | `text` / `uuid` | Chat session identifier |
| `role` | `text` | `user` or `assistant` |
| `content` | `text` | Message content |
| `created_at` | `timestamptz` | Default: `now()` |

Example SQL script:

```sql
create table chat_history (
  id bigint generated by default as identity primary key,
  session_id text not null,
  role text not null,
  content text not null,
  created_at timestamptz default now()
);
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

### Via GitHub Codespaces / Dev Container

The project includes a `.devcontainer` configuration. In a Codespace, the app starts automatically on port `8501` after dependencies are installed.

---

## 📖 Usage

1. **Choose a mode** in the sidebar:
   - **💬 Classic Chat** — Chat freely with the assistant.
   - **📄 Document Analysis** — Upload a `.txt` or `.pdf` file, then ask questions about its content.
2. **Ask a question** using the input bar at the bottom of the screen.
3. **Watch the agent work** — the status box shows each reasoning step and tool call live.
4. **Reset** the conversation with the *🗑️ Recommencer la discussion* button.

### Example prompts

- *"What was the score of Real Madrid's last match?"* → triggers web search.
- *"What's the weather in Paris?"* → triggers the weather tool.
- *"What is (45 * 12) / 3?"* → triggers the calculator.
- In document mode: *"Summarize this document"*, then *"and in English?"*.

---

## 🔒 Security

- Secrets (`GEMINI_API_KEY`, Supabase credentials) are stored in `.streamlit/secrets.toml`, excluded from version control.
- The `calculatrice` function uses `eval` with `__builtins__` disabled to limit arbitrary code execution risks.

---

## 📄 License

This project is provided for learning purposes. Add a license if you intend to distribute it.