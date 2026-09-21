import os
from uuid import uuid4

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("RAG_LLM_DEVICE", "cpu")

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.services.embedding_service import embedding_service
from app.models.schemas import ChatRequest, SearchRequest, TextCreateRequest
from app.services.chunking_service import chunking_service
from app.services.pdf_service import pdf_service
from app.services.rag_service import rag_service
from app.services.vector_service import VECTOR_SIZE, vector_service

app = FastAPI(
    title="RAG Knowledge Assistant",
    description="RAG API using FastAPI, Hugging Face and Qdrant",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "RAG Knowledge Assistant API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "rag-api",
    }


@app.get("/qdrant/health")
def qdrant_health():
    return vector_service.health()


@app.post("/qdrant/collection")
def create_qdrant_collection():
    return vector_service.create_collection() 


@app.get("/embedding/test")
def test_embedding():
    text = "React Native is used for mobile application development."

    embedding = embedding_service.generate(text)

    return {
        "text": text,
        "dimensions": len(embedding),
        "preview": embedding[:5],
    }


@app.post("/vectors/add")
def add_vector(request: TextCreateRequest):
    embedding = embedding_service.generate(request.text)

    if len(embedding) != VECTOR_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dimension must be {VECTOR_SIZE}; received {len(embedding)}.",
        )

    return vector_service.add_text(text=request.text, vector=embedding)


@app.post("/vectors/search")
def search_vectors(request: SearchRequest):
    embedding = embedding_service.generate(request.query)

    if len(embedding) != VECTOR_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dimension must be {VECTOR_SIZE}; received {len(embedding)}.",
        )

    return {
        "query": request.query,
        "results": vector_service.search(vector=embedding, limit=request.limit),
    }


async def read_pdf_upload(file: UploadFile) -> bytes:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    max_size = 20 * 1024 * 1024
    if len(pdf_bytes) > max_size:
        raise HTTPException(status_code=413, detail="PDF size must be 20 MB or less.")

    return pdf_bytes


@app.post("/documents/extract")
async def extract_document(file: UploadFile = File(...)):
    pdf_bytes = await read_pdf_upload(file)

    try:
        result = pdf_service.extract_text(pdf_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process PDF: {str(exc)}",
        ) from exc

    return {
        "status": "extracted",
        "file_name": file.filename,
        **result,
    }


@app.post("/documents/chunk-preview")
async def chunk_preview(file: UploadFile = File(...)):
    pdf_bytes = await read_pdf_upload(file)

    try:
        extraction = pdf_service.extract_text(pdf_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process PDF: {str(exc)}",
        ) from exc

    chunks = chunking_service.chunk_pages(extraction["pages"])
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    return {
        "status": "chunked",
        "file_name": file.filename,
        "page_count": extraction["page_count"],
        "chunk_count": len(chunks),
        "chunk_size": chunking_service.chunk_size,
        "chunk_overlap": chunking_service.chunk_overlap,
        "chunks": chunks,
    }


@app.post("/documents/index")
async def index_document(file: UploadFile = File(...)):
    pdf_bytes = await read_pdf_upload(file)

    try:
        extraction = pdf_service.extract_text(pdf_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process PDF: {str(exc)}",
        ) from exc

    chunks = chunking_service.chunk_pages(extraction["pages"])
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    document_id = str(uuid4())
    embeddings = embedding_service.generate_many([chunk["text"] for chunk in chunks])

    if len(embeddings) != len(chunks):
        raise HTTPException(
            status_code=500,
            detail="Embedding count does not match chunk count.",
        )

    if any(len(embedding) != VECTOR_SIZE for embedding in embeddings):
        raise HTTPException(
            status_code=500,
            detail=f"Each embedding must have {VECTOR_SIZE} dimensions.",
        )

    try:
        indexing_result = vector_service.index_document_chunks(
            document_id=document_id,
            file_name=file.filename,
            chunks=chunks,
            embeddings=embeddings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "indexed",
        "document_id": document_id,
        "file_name": file.filename,
        "page_count": extraction["page_count"],
        "chunk_count": len(chunks),
        **indexing_result,
    }


@app.get("/documents")
def list_documents():
    documents = vector_service.list_documents()
    return {
        "documents": documents,
        "total_documents": len(documents),
    }


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    result = vector_service.delete_document(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return result


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not complete RAG request: {exc}",
        ) from exc
