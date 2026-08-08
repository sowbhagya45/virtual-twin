# virtual-twin

An agentic AI virtual twin of **Sowbhagya Mohanthy** — built with LangGraph, Gemini, and Streamlit.

It answers questions about his skills, experience, and projects using RAG over his resume and deep profile,
books meetings via a multi-step scheduler agent, and notifies him when someone asks something outside
the knowledge base.

---

## Architecture

```
User (Streamlit chat)
        │
        ▼
  ┌─────────────────────────────────────┐
  │  SUPERVISOR (Gemini 2.0 Flash Lite) │  ← classifies intent (keyword-first, LLM fallback)
  │  routes to one of four sub-agents   │
  └──────────┬──────────────────────────┘
             │
    ┌────────┼────────────────────────┐
    │        │                        │
    ▼        ▼          ▼             ▼
 RAG       Scheduler  Notifier    Chitchat
 Agent     Agent      Agent       Agent
   │
   │ (knowledge_gap?)
   ▼
 Notifier Agent
```

| Agent | LLM | Job |
|---|---|---|
| Supervisor | Gemini (configurable) | Intent routing (keyword-first = zero tokens) |
| RAG Agent | Gemini (configurable) | Retrieves from ChromaDB, answers profile questions |
| Scheduler Agent | Gemini (configurable) | Multi-turn meeting booking flow |
| Notifier Agent | Gemini (configurable) | Captures unknowns, emails Sowbhagya |
| Chitchat Agent | Gemini (configurable) | Small talk with Sowbhagya's personality |

Default model: **`gemini-3.5-flash`** (fresh free-tier quota bucket).
Override via `GEMINI_MODEL` env var, or use the **🤖 Model** selector in the sidebar to switch on quota exhaustion.

State is persisted per browser session with **InMemorySaver** (zero infra — works on Streamlit Cloud).

---

## Knowledge Base

Three sources are merged and indexed on first run:

| Source | Content |
|---|---|
| `data/resume.pdf` | 2-page PDF resume |
| `data/interview_context.txt` | 571-line deep profile — projects, philosophy, tech comparisons |
| `EXTRA_KNOWLEDGE` (inline in `ingest.py`) | FAQ, bio, contact details, availability |

**161 chunks** total · embedded with `gemini-embedding-001` · stored in ChromaDB

---

## Project Structure

```
virtual-twin/
├── app.py                        # Streamlit entry point
├── ingest.py                     # Knowledge base builder (run once)
├── requirements.txt              # Streamlit Community Cloud dependencies
├── pyproject.toml                # uv / local dev project file
├── .env.example                  # Copy to .env for local dev
├── .streamlit/
│   ├── config.toml               # Streamlit theme + server config
│   └── secrets.toml.example      # Copy to secrets.toml for local dev
├── agents/
│   ├── __init__.py
│   ├── graph.py                  # LangGraph supervisor + 4 ReAct sub-agents
│   ├── tools.py                  # retrieve_profile_info, book_meeting, notify_owner
│   └── prompts.py                # All system prompts (grounded, short, token-efficient)
└── data/
    ├── resume.pdf                # Source 1: resume PDF
    └── interview_context.txt     # Source 2: deep profile document
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Google AI Studio API key: [aistudio.google.com](https://aistudio.google.com)

### 1. Clone and install

```bash
git clone https://github.com/sowbhagya/virtual-twin.git
cd virtual-twin

# With uv (recommended — fast, reproducible)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY at minimum
```

### 3. Build the knowledge base

```bash
uv run python ingest.py
# or: python ingest.py
```

This embeds **3 sources** (resume PDF + `interview_context.txt` + inline FAQ) into `./chroma_db`
using `gemini-embedding-001`.

> **Free-tier note:** Gemini free tier allows 100 embed requests/minute.  
> The script throttles at 0.7s per chunk — 161 chunks take ~2 minutes.  
> Re-run whenever you update the resume or deep profile.

### 4. Run locally

```bash
uv run streamlit run app.py
# or: streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Streamlit Community Cloud Deployment

1. Push this repo to GitHub (ensure `data/resume.pdf` and `data/interview_context.txt` are committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, file `app.py`.
4. Open **Advanced settings → Secrets** and add:

```toml
GOOGLE_API_KEY = "AIza..."
SENDGRID_API_KEY = "SG...."        # optional — enables email notifications
OWNER_EMAIL = "sudhirkumar02001@gmail.com"
LANGSMITH_API_KEY = "ls__..."      # optional — enables LangSmith tracing
```

5. Click **Deploy**.

> **First boot:** If `chroma_db/` is not committed to the repo, `app.py` auto-runs `ingest.py`
> on first boot. This takes **~2 minutes** due to Gemini free-tier rate limits.
>
> **Recommended:** Commit `chroma_db/` to the repo (remove it from `.gitignore` or run
> `git add -f chroma_db/`) so cold starts are instant.

---

## UI Features

| Feature | Details |
|---|---|
| Quick-action chips | 4 one-click prompt buttons for common queries |
| Download resume | Sidebar button downloads the PDF directly |
| Agent badges | Shows which agent handled each response |
| Streaming text | Typewriter effect on all AI responses |
| Knowledge-gap banner | Info banner when a question was outside the KB |
| Active agent indicator | Sidebar shows which agent handled the last turn |

---

## Adding more knowledge

1. Edit the `EXTRA_KNOWLEDGE` string in `ingest.py` — add Q&A, new projects, or bio updates.
2. Or add more files to `data/` and load them in `ingest.py`.
3. Re-run `python ingest.py` to rebuild the vector store.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Agent framework | LangGraph 0.2+ (supervisor + ReAct sub-agents) |
| LLM | Gemini 3.5-flash by default; any Gemini model via `GEMINI_MODEL` |
| Embeddings | `gemini-embedding-001` |
| Vector store | ChromaDB (persisted locally / in repo) |
| Memory / checkpointing | `InMemorySaver` (per-session, zero infra) |
| Notifications | SendGrid (optional — logs to console if not configured) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud (free) |
| Observability | LangSmith (optional) |
| Package manager | uv |
