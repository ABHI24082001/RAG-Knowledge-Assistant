from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import get_settings


COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 384


class VectorService:
    def __init__(self):
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is not None:
            return self._client

        settings = get_settings()
        if settings.qdrant_mode == "local":
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(settings.qdrant_path))
        else:
            if not settings.qdrant_url:
                raise RuntimeError("QDRANT_URL is required when QDRANT_MODE is url or cloud.")
            if settings.qdrant_mode == "cloud" and not settings.qdrant_api_key:
                raise RuntimeError("QDRANT_API_KEY is required when QDRANT_MODE is cloud.")
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=settings.qdrant_timeout_seconds,
            )
        return self._client

    def create_collection(self):
        exists = self.client.collection_exists(
            collection_name=COLLECTION_NAME
        )

        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

            return {
                "status": "created",
                "collection": COLLECTION_NAME,
            }

        return {
            "status": "already_exists",
            "collection": COLLECTION_NAME,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def health(self):
        collections = self.client.get_collections()

        return {
            "status": "connected",
            "collections": [
                collection.name
                for collection in collections.collections
            ],
        }

    def add_text(
        self,
        text: str,
        vector: list[float],
        metadata: dict | None = None,
    ):
        point_id = str(uuid4())
        payload = {**(metadata or {}), "text": text}

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
            wait=True,
        )

        return {
            "status": "stored",
            "id": point_id,
            "text": text,
        }

    def search(self, vector: list[float], limit: int = 3):
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "text": (point.payload or {}).get("text", ""),
                "source_type": (point.payload or {}).get("source_type"),
                "document_id": (point.payload or {}).get("document_id"),
                "file_name": (point.payload or {}).get("file_name"),
                "page": (point.payload or {}).get("page"),
                "chunk_id": (point.payload or {}).get("chunk_id"),
            }
            for point in response.points
        ]

    def search_documents(
        self,
        vector: list[float],
        limit: int = 3,
        document_id: str | None = None,
    ):
        conditions = [
            FieldCondition(
                key="source_type",
                match=MatchValue(value="document"),
            )
        ]
        if document_id:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "text": (point.payload or {}).get("text", ""),
                "source_type": (point.payload or {}).get("source_type"),
                "document_id": (point.payload or {}).get("document_id"),
                "file_name": (point.payload or {}).get("file_name"),
                "page": (point.payload or {}).get("page"),
                "chunk_id": (point.payload or {}).get("chunk_id"),
            }
            for point in response.points
        ]

    @staticmethod
    def _document_filter(document_id: str | None = None) -> Filter:
        conditions = [
            FieldCondition(
                key="source_type",
                match=MatchValue(value="document"),
            )
        ]
        if document_id:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )
        return Filter(must=conditions)

    def _document_points(self, document_id: str | None = None) -> list:
        points = []
        offset = None
        document_filter = self._document_filter(document_id)

        while True:
            batch, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=document_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                break

        return points

    def list_documents(self) -> list[dict]:
        documents = {}
        for point in self._document_points():
            payload = point.payload or {}
            document_id = payload.get("document_id")
            if not document_id:
                continue

            document = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "file_name": payload.get("file_name"),
                    "pages": set(),
                    "chunk_count": 0,
                },
            )
            if payload.get("page") is not None:
                document["pages"].add(payload["page"])
            document["chunk_count"] += 1

        return sorted(
            [
                {
                    "document_id": document["document_id"],
                    "file_name": document["file_name"],
                    "page_count": max(document["pages"], default=0),
                    "chunk_count": document["chunk_count"],
                }
                for document in documents.values()
            ],
            key=lambda document: (document["file_name"] or "", document["document_id"]),
        )

    def delete_document(self, document_id: str) -> dict | None:
        points = self._document_points(document_id)
        if not points:
            return None

        file_name = (points[0].payload or {}).get("file_name")
        deleted_chunks = len(points)
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=self._document_filter(document_id),
            wait=True,
        )
        return {
            "status": "deleted",
            "document_id": document_id,
            "file_name": file_name,
            "deleted_chunks": deleted_chunks,
        }

    def index_document_chunks(
        self,
        document_id: str,
        file_name: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")
        if any(len(embedding) != VECTOR_SIZE for embedding in embeddings):
            raise ValueError(f"Each embedding must have {VECTOR_SIZE} dimensions.")

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "source_type": "document",
                    "document_id": document_id,
                    "file_name": file_name,
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "characters": chunk["characters"],
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        return {"indexed_chunks": len(points)}


vector_service = VectorService()
