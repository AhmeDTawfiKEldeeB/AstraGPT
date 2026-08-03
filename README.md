<p align="center">
  <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuD55I3cOGAeqaf_zriVqnIJa9yt0aFRZ0qnbUJKSQFdaoy9tgmwZBmrKJWXMIFeCTmV4tQvY2vqumvofkyC0sf7ISM8SIkF4fgAM5Ntj4rUQ5p2oDvB7Jk3iYLQvAInjr4VctngHOrW8pJ4AsYRVJjhYjafapgYTIhUFbO_2my-un6YWbbVRmYCGaWo7GyH1Wb5_8Wa5g38cj1gkuKs-N05UW68ZnKqhyW0aUX8wDNDRpBTLoPecOce" alt="AstraGPT" width="160">
</p>

<h1 align="center">AstraGPT</h1>

<p align="center">
  <b>A production-style, agentic AI assistant with RAG, live web search, long-term memory, and streaming responses.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-1.2-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/Google%20Gemini-LLM-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-D800FF?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant">
  <img src="https://img.shields.io/badge/SQLite-Storage-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

AstraGPT is a full-stack ChatGPT-style assistant that doesn't just chat — it *acts*. It searches the web in real time, retrieves answers from documents you upload, remembers facts about you across sessions, computes math, checks the weather, and streams every token live to a polished, Material-Design-3 UI.

---

## Why AstraGPT? ✨

- **🤖 Truly agentic** — a LangGraph state machine decides *when* and *which* tool to use, not a scripted flow.
- **📄 Your documents, your answers** — upload a PDF, DOCX, TXT, Markdown, Python, or CSV and ask questions against its content (RAG).
- **🌐 Live knowledge** — time-sensitive questions are answered with fresh web search, not stale training data.
- **🧠 It remembers you** — ask it to remember a preference or fact and it recalls it later, per conversation.
- **⚡ Real-time streaming** — tokens stream over SSE as they're generated, with live tool-execution indicators in the UI.
- **🔒 Zero external infrastructure** — SQLite + on-disk Qdrant mean the whole stack runs locally with one command.

---

## Features 🚀

| Capability | How it works |
|---|---|
| 💬 Multi-turn chat with memory | LangGraph checkpointer (SQLite) persists every thread |
| ⚡ Streaming responses | SSE events: `token`, `tool_start`, `tool_end`, `done`, `error` |
| 🌐 Web search | Tavily Search — triggered automatically for time-sensitive questions |
| 📄 Document Q&A (RAG) | Upload → chunk → embed (Gemini) → semantic retrieval from Qdrant |
| 🧠 Long-term memory | `remember_this` / `recall_memory` tools backed by SQLite |
| 🧮 Math & calculation | SymPy-powered `calculator` tool |
| ⛅ Weather | OpenWeatherMap tool |
| 🧬 Multi-model support | Switch Gemini models from the UI (5 presets, safely allow-listed) |
| 🕘 Conversation history | Searchable sidebar with auto-generated titles |
| 🌗 Dark / light theme | Material Design 3 color system, persisted preference |
| 📎 File attachments | Multiple files per chat with upload progress chips |
| 🎤 Voice input | Microphone capture in the browser |
| 📋 Copy / regenerate | Per-message actions on assistant replies |

---

## Architecture 🏗️

```
┌─────────────────────────────┐
│  Frontend (Vanilla JS +     │
│  Tailwind, Material 3 UI)   │
└──────────────┬──────────────┘
               │  fetch / SSE
┌──────────────▼──────────────┐
│  FastAPI  (app.py)          │
│  /chat/stream  /upload      │
│  /conversations /history    │
└──────────────┬──────────────┘
               │ LangGraph StateGraph
┌──────────────▼──────────────────────────────────────┐
│  Agent (LangGraph + Gemini)                         │
│  ┌──────────┐  conditional   ┌──────────┐           │
│  │ chatbot  │ ─────────────▶ │  tools   │           │
│  └──────────┘ ◀───────────── └──────────┘           │
│  Checkpointer: SQLite (data/langgraph_checkpoints)  │
└──────────────┬──────────────────────────────────────┘
               │
   ┌───────────┼──────────────────────────┐
   ▼           ▼           ▼              ▼
Tavily      OpenWeather  SymPy        Qdrant (local, on-disk)
web search  weather     calculator    vector store
                                          ▲
                              RAG pipeline: extract → split →
                              embed (gemini-embedding-001) →
                              cosine similarity search

   SQLite (data/chatbot_memory.db)
   └─ conversations, chat_messages, long_term_memory
```

**Agent loop:** the LLM calls tools when needed (`tools_condition`), tool results feed back into the chat node, and every step is checkpointed so the conversation graph can be resumed at any time.

---

## Tech Stack 🛠️

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| Agent framework | LangGraph, LangChain, LangChain-Groq |
| LLM | Google Gemini (`ChatGoogleGenerativeAI`), 5 selectable models |
| Embeddings | `gemini-embedding-001` |
| Vector store | Qdrant (local, on-disk, cosine distance) |
| Web search | Tavily |
| Math | SymPy |
| Storage | SQLite (chat history, memory, graph checkpoints) via SQLAlchemy |
| File parsing | PyPDF, docx2txt |
| Frontend | Vanilla HTML/CSS/JS, Tailwind CSS, Material Design 3 theming |

---

## Getting Started 🚀

### Prerequisites 📋

- Python **3.12+**
- API keys (see [Environment Variables](#environment-variables))

### 1. Clone & install 📦

```bash
git clone https://github.com/<your-user>/AstraGPT.git
cd AstraGPT

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment 🔑

```bash
cp .env.example .env   # or create .env manually
```

### 3. Run the server ▶️

```bash
uvicorn app:app --reload
```

Open **http://localhost:8000** — the frontend is served automatically.

![AstraGPT UI](https://lh3.googleusercontent.com/aida-public/AB6AXuDE1wigTtUIxQPPUOik94VZx2Qh10Eb8gEzcT2ThjJh1Q8CRlVqJoZJKzOtkynbPM7_eRDTpecSAkbbZ3cDk4tgoneLuMUwiVt8gMpJT4HettH0YH74iOKhXbMCT3IM5OG9vpM1tJRKiKkfAH0eRgEocSU9zi9JmuHkf2fKX7GzUiJ3Hjw4dLNj8LaSOyevr0GX_w4OA9ntIiHbb0l2rp6hkC5nIZasmH3AVewJ4c95tloFq6NoDpEKDgCEnSn2-V-l7Q" width="100%">

---

## Environment Variables 🔐

Create a `.env` file in the project root:

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | **Yes** | Google AI Studio key for Gemini LLM + embeddings |
| `TAVILY_API_KEY` | **Yes** | Tavily key for web-search tool |
| `OPENWEATHER_API_KEY` | **Yes** | OpenWeatherMap key for the weather tool |
| `GOOGLE_MODEL` | No | Default Gemini model (default: `gemini-3.5-flash`) |
| `LANGSMITH_TRACING` | No | Set `true` to enable LangSmith observability |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_ENDPOINT` | No | LangSmith endpoint |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

---

## Selectable Models 🧬

The model picker in the UI is safely allow-listed in `src/Services/Agent/agent.py`:

| Model | Notes |
|---|---|
| `gemini-3.6-flash` | Latest fast model |
| `gemini-3.5-flash` | Default |
| `gemini-3.1-flash-lite` | Lightweight & fast |


---

## API Reference 🔌

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat/stream` | Send a message, receive an **SSE** stream of tokens and tool events |
| `POST` | `/upload` | Upload a document (`multipart/form-data`) to the RAG knowledge base |
| `GET` | `/conversations` | List all conversations (newest first) |
| `GET` | `/history/{thread_id}` | Full message history for a thread |
| `GET` | `/` | Serve the frontend (static mount) |

**Supported uploads:** `.pdf`, `.docx`, `.txt`, `.md`, `.py`, `.csv`

**SSE event types:**

```
event: token       data: {"content": "..."}
event: tool_start  data: {"tool": "tavily_search"}
event: tool_end    data: {"tool": "tavily_search"}
event: done        data: {}
event: error       data: {"message": "..."}
```

---

## Project Structure 📂

```
AstraGPT/
├── app.py                          # FastAPI app: routes, SSE streaming, uploads
├── main.py                         # CLI entry point
├── requirements.txt                # Python dependencies
├── .env                            # Secrets (never commit)
├── frontend/                       # Vanilla JS UI (Tailwind + Material 3)
│   ├── index.html
│   ├── script.js
│   └── style.css
├── src/
│   ├── Services/
│   │   ├── Agent/
│   │   │   ├── agent.py            # LangGraph workflow + Gemini + checkpointer
│   │   │   └── tools.py            # Tool definitions (Tavily, weather, calc, RAG, memory)
│   │   └── Rag/
│   │       └── rag_service.py      # Extraction → chunking → embedding → retrieval
│   └── infrastructure/
│       └── sqlalchemy_database.py  # Conversations, chat messages, long-term memory
├── uploads/                        # Uploaded documents (git-ignored)
├── data/                           # SQLite DBs (git-ignored)
└── qdrant_db/                      # Local Qdrant store (git-ignored)
```

---

## How the Agent Thinks 🧭

The system prompt (`agent.py`) teaches the model when to act:

- **🗞️ "What's the latest news / today / current prices?"** → Tavily web search
- **📄 "Anything about my uploaded file?"** → RAG document retrieval
- **🧠 "Remember that I prefer X"** → long-term memory tool
- **🔎 "What did I tell you before?"** → recall memory tool
- **🧮 Math questions** → SymPy calculator
- **⛅ Weather questions** → OpenWeatherMap tool

---

## Roadmap 🗺️

- [ ] Authentication & multi-user workspaces
- [ ] Docker + docker-compose deployment
- [ ] More chunking strategies & hybrid (BM25 + vector) retrieval
- [ ] Speech-to-text backend (beyond browser mic)
- [ ] Streaming progress for uploads
- [ ] Unit tests for RAG pipeline and agent tools

---

## License ⚖️

This project is licensed under the terms of the [LICENSE](LICENSE) file.
