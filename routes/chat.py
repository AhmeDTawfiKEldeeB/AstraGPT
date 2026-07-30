import asyncio
import json
import threading

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.infrastructure.sqlalchemy_database import (
    create_or_update_conversation,
    save_chat_message,
)
from src.Services.Agent.agent import get_agent
from src.Services.Agent.tools import set_current_thread_id

router = APIRouter()


@router.post("/stream")
async def chat_stream(request: Request):
    body = await request.json()
    thread_id = body["thread_id"]
    message = body["message"]
    model = body.get("model", "gemini-2.5-flash")

    set_current_thread_id(thread_id)

    save_chat_message(thread_id, "user", message)
    create_or_update_conversation(thread_id, message)

    async def event_generator():
        agent = get_agent(model)
        config = {"configurable": {"thread_id": thread_id}}
        input_data = {"messages": [{"role": "user", "content": message}]}

        full_response = ""
        queue = asyncio.Queue()
        sent_tool_starts = set()

        def stream_in_thread():
            try:
                for msg_chunk, metadata in agent.stream(
                    input_data, config, stream_mode="messages"
                ):
                    if msg_chunk.content:
                        queue.put_nowait(("token", msg_chunk.content))

                    tool_chunks = getattr(msg_chunk, "tool_call_chunks", None)
                    if tool_chunks:
                        for tc in tool_chunks:
                            name = tc.get("name") or tc.get("id", "")
                            if name and name not in sent_tool_starts:
                                sent_tool_starts.add(name)
                                queue.put_nowait(("tool_start", name))

                    msg_type = getattr(msg_chunk, "type", "")
                    msg_name = getattr(msg_chunk, "name", "")
                    if msg_type == "tool" or msg_name:
                        if msg_name:
                            queue.put_nowait(("tool_end", msg_name))

                queue.put_nowait(("done", full_response))
            except Exception as e:
                queue.put_nowait(("error", str(e)))

        thread = threading.Thread(target=stream_in_thread, daemon=True)
        thread.start()

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
