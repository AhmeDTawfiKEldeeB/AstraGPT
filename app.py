from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routes.chat import router as chat_router
from routes.conversations import router as conversations_router
from routes.upload import router as upload_router
from src.infrastructure.sqlalchemy_database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AstraGPT", lifespan=lifespan)

app.include_router(chat_router, prefix="/chat")
app.include_router(conversations_router, prefix="/conversations")
app.include_router(upload_router, prefix="/upload")

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
