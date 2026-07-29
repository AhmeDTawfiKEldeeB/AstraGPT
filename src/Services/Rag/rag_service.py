from pathlib import Path
from typing import List
from uuid import uuid4

import docx2txt
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance,FieldCondition,Filter,MatchValue,PointStruct,VectorParams,)

load_dotenv()

UPLOADS_DIR = Path("uploads")
DB_PATH = "qdrant_db"
COLLECTION_NAME = "knowledge_base"

UPLOADS_DIR.mkdir(exist_ok=True)
Path(DB_PATH).mkdir(exist_ok=True)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
client = QdrantClient(path=DB_PATH)

VECTOR_SIZE = len(
    embeddings.embed_query("test")
)


# Read file
def read_file(file_path: str) -> str:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md", ".py", ".csv"}:
        return file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    if suffix == ".pdf":
        reader = PdfReader(file_path)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    if suffix == ".docx":
        return docx2txt.process(str(file_path))

    raise ValueError(f"Unsupported file type: {suffix}")

# Chunk text
def chunk_text(text: str,source: str,chunk_size: int = 800,chunk_overlap: int = 200,) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",". "," ","",],)
    
    return splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}],
    )


# Create collection
def create_collection():
    collections = client.get_collections().collections
    if COLLECTION_NAME in [c.name for c in collections]:
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
# Embed documents
def embed_documents(documents: List[Document],) -> List[PointStruct]:
    points = []
    for document in documents:
        embedding = embeddings.embed_query(
            document.page_content
        )

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "text": document.page_content,
                    **document.metadata,
                },
            )
        )

    return points

# Store document
def store_document(file_path: str,thread_id: str,):
    create_collection()
    text = read_file(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")
    documents = chunk_text(text=text,source=Path(file_path).name,)
    for document in documents:
        document.metadata["thread_id"] = thread_id

    points = embed_documents(documents)
    client.upsert(collection_name=COLLECTION_NAME,points=points,)
    return {
        "filename": Path(file_path).name,
        "chunks": len(points),
    }

# Retrieve context
def retrieve_context(query: str,thread_id: str,top_k: int = 4,) -> str:

    query_vector = embeddings.embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=Filter(
            must=[FieldCondition(key="thread_id",match=MatchValue(value=thread_id),)]),
    ).points

    if not results:
        return "No relevant uploaded document content found."
    context = []
    for i, point in enumerate(results, start=1):
        source = point.payload.get("source","uploaded document",)
        context.append(
            f"[Source {i}: {source}]\n"
            f"{point.payload['text']}"
        )
    return "\n\n".join(context)