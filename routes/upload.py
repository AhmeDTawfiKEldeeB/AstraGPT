from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form

from src.Services.Rag.rag_service import store_document, read_file

router = APIRouter()
UPLOADS_DIR = Path("uploads")


@router.post("")
async def upload_file(file: UploadFile = File(...), thread_id: str = Form("default")):
    UPLOADS_DIR.mkdir(exist_ok=True)

    file_path = UPLOADS_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    result = store_document(str(file_path), thread_id)

    return {
        "filename": file.filename,
        "chunks": result["chunks"],
        "thread_id": thread_id,
    }
