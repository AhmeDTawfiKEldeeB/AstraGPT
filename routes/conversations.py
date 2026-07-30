from fastapi import APIRouter
from pydantic import BaseModel

from src.infrastructure.sqlalchemy_database import (
    get_chat_history,
    list_conversations,
)

router = APIRouter()


@router.get("")
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


@router.get("/{thread_id}")
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
