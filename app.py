from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

import asyncio
import json
import threading

from langchain_core.messages import AIMessageChunk, ToolMessageChunk, ToolMessage

from src.infrastructure.sqlalchemy_database import init_db, get_chat_history, save_chat_message, create_or_update_conversation, list_conversations
from src.Services.Agent.agent import get_agent
from src.Services.Agent.tools import set_current_thread_id
from src.Services.Rag.rag_service import store_document


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AstraGPT", lifespan=lifespan)



ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}


def event_generator_stream(model: str, thread_id: str, message: str):
    """Synchronous generator running in a thread."""
    agent = get_agent(model)
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {"messages": [{"role": "user", "content": message}]}

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

    yield ("done", full_response)


@app.post("/chat/stream")
async def chat_stream(body: dict):
    thread_id = body["thread_id"]
    message = body["message"]
    model = body.get("model", "llama-3.3-70b-versatile")

    set_current_thread_id(thread_id)

    save_chat_message(thread_id, "user", message)
    create_or_update_conversation(thread_id, message)

    async def event_generator():
        queue = asyncio.Queue()
        sent_tool_starts = set()

        def stream_in_thread():
            try:
                for event_type, data in event_generator_stream(model, thread_id, message):
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