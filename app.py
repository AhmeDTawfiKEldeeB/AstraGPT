from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

import asyncio
import base64
import json
import os
import threading

import requests
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessageChunk, ToolMessageChunk, ToolMessage

from src.infrastructure.sqlalchemy_database import init_db, get_chat_history, save_chat_message, create_or_update_conversation, list_conversations
from src.Services.Agent.agent import get_agent, DEFAULT_MODEL
from src.Services.Agent.tools import set_current_thread_id, tavily_tool
from src.Services.Rag.rag_service import store_document, retrieve_context


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AstraGPT", lifespan=lifespan)



ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}


TIME_SENSITIVE_KEYWORDS = (
    "news", "latest", "current", "today", "now", "recent", "update",
    "price", "prices", "stock", "stocks", "weather", "forecast",
    "release", "releases", "released", "election", "score", "scores",
    "who won", "who became", "who is the new", "breaking",
)


def is_time_sensitive(message: str) -> bool:
    lowered = message.lower()
    return any(kw in lowered for kw in TIME_SENSITIVE_KEYWORDS)


def event_generator_stream(model: str, thread_id: str, message: str, uploaded_files: list[str] | None = None):
    """Synchronous generator running in a thread."""
    agent = get_agent(model)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}

    user_content = message
    web_searched = False

    if uploaded_files:
        files_list = ", ".join(uploaded_files)
        doc_context = retrieve_context(query=message, thread_id=thread_id, top_k=6)
        user_content = (
            f"[Uploaded document(s): {files_list}]\n"
            f"--- DOCUMENT CONTENT (for reference only) ---\n"
            f"{doc_context}\n"
            f"--- END DOCUMENT CONTENT ---\n\n"
            f"{message}"
        )

    if is_time_sensitive(message):
        try:
            search_results = tavily_tool.invoke(message)
            if search_results and "Error" not in str(search_results)[:200]:
                yield ("tool_start", "tavily_search")
                web_searched = True
                user_content = (
                    f"[Web search results for the user's question]\n"
                    f"{search_results}\n\n"
                    f"{message}"
                )
        except Exception:
            pass

    input_data = {"messages": [{"role": "user", "content": user_content}]}

    full_response = ""
    for msg_chunk, metadata in agent.stream(input_data, config, stream_mode="messages"):
        if isinstance(msg_chunk, AIMessageChunk):
            if msg_chunk.content:
                text = msg_chunk.content
                if isinstance(text, list):
                    text = "".join(
                        b.get("text", "")
                        for b in text if isinstance(b, dict)
                    )
                if text:
                    full_response += text
                    yield ("token", text)

            for tc in (msg_chunk.tool_call_chunks or []):
                name = tc.get("name") or ""
                if name:
                    yield ("tool_start", name)

        if isinstance(msg_chunk, (ToolMessage, ToolMessageChunk)):
            name = getattr(msg_chunk, "name", None) or ""
            if name:
                yield ("tool_end", name)

    if web_searched:
        yield ("tool_end", "tavily_search")

    yield ("done", full_response)


@app.post("/chat/stream")
async def chat_stream(body: dict):
    thread_id = body["thread_id"]
    message = body["message"]
    model = body.get("model", DEFAULT_MODEL)
    uploaded_files = body.get("uploaded_files") or []

    set_current_thread_id(thread_id)

    save_chat_message(thread_id, "user", message)
    create_or_update_conversation(thread_id, message)

    async def event_generator():
        queue = asyncio.Queue()
        sent_tool_starts = set()

        def stream_in_thread():
            try:
                for event_type, data in event_generator_stream(model, thread_id, message, uploaded_files):
                    queue.put_nowait((event_type, data))
            except Exception as e:
                queue.put_nowait(("error", str(e)))

        thread = threading.Thread(target=stream_in_thread, daemon=True)
        thread.start()

        full_response = ""
        while True:
            event_type, data = await queue.get()
            if event_type == "done":
                full_response = data
                break
            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'message': data})}\n\n"
                return
            elif event_type == "token":
                full_response += data
                yield f"event: token\ndata: {json.dumps({'content': data})}\n\n"
            elif event_type == "tool_start":
                if data not in sent_tool_starts:
                    sent_tool_starts.add(data)
                    yield f"event: tool_start\ndata: {json.dumps({'tool': data})}\n\n"
            elif event_type == "tool_end":
                yield f"event: tool_end\ndata: {json.dumps({'tool': data})}\n\n"

        if full_response:
            save_chat_message(thread_id, "assistant", full_response)

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), thread_id: str = Form("default")):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    try:
        result = store_document(str(file_path), thread_id)
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "filename": file.filename,
        "chunks": result["chunks"],
        "thread_id": thread_id,
    }


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    mime_type = file.content_type or "audio/webm"
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set")

    base64_audio = base64.b64encode(content).decode("ascii")

    prompt = (
        "Transcribe this audio exactly as spoken. "
        "Keep the original language of the speaker (detect it automatically). "
        "Do not add punctuation that was not spoken. "
        "Reply with the transcription text only."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": base64_audio}},
                    {"text": prompt},
                ]
            }
        ]
    }

    models = [os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"), "gemini-2.0-flash"]
    last_error = None
    for model in dict.fromkeys(models):
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": api_key},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                parts = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [])
                )
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    return {"text": text, "model": model}
                raise HTTPException(status_code=422, detail="No speech detected in the audio")
            last_error = f"Gemini API returned HTTP {resp.status_code}: {resp.text[:300]}"
        except HTTPException:
            raise
        except Exception as e:
            last_error = str(e)

    raise HTTPException(status_code=502, detail=f"Speech-to-text failed: {last_error}")


@app.get("/conversations")
def get_conversations():
    conversations = list_conversations()
    return [
        {
            "thread_id": c.thread_id,
            "title": c.title,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in conversations
    ]


@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    messages = get_chat_history(thread_id)
    return [
        {
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
    ]


# Serve frontend
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")