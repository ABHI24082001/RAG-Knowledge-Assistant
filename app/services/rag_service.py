import logging

from app.config import get_settings
from app.services.embedding_service import embedding_service
from app.services.llm_service import NO_ANSWER, llm_service
from app.services.vector_service import VECTOR_SIZE, vector_service

logger = logging.getLogger(__name__)


class RAGService:
    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        document_id: str | None = None,
    ) -> dict:
        embedding = embedding_service.generate(question)
        if len(embedding) != VECTOR_SIZE:
            raise ValueError(
                f"Question embedding must have {VECTOR_SIZE} dimensions; received {len(embedding)}."
            )

        retrieved = vector_service.search_documents(
            vector=embedding,
            limit=top_k,
            document_id=document_id,
        )

        threshold = get_settings().rag_min_relevance_score
        relevant_chunks = [
            chunk for chunk in retrieved if chunk["score"] >= threshold
        ]
        logger.info(
            "RAG retrieval document_id=%s threshold=%.3f retrieved_scores=%s relevant_scores=%s",
            document_id or "all",
            threshold,
            [round(chunk["score"], 4) for chunk in retrieved],
            [round(chunk["score"], 4) for chunk in relevant_chunks],
        )

        unique_chunks = []
        seen = set()
        for chunk in relevant_chunks:
            identity = (chunk["document_id"], chunk["chunk_id"])
            if identity not in seen:
                seen.add(identity)
                unique_chunks.append(chunk)

        if not unique_chunks:
            return {
                "question": question,
                "answer": NO_ANSWER,
                "sources": [],
                "retrieved_chunks": 0,
            }

        context = "\n\n".join(
            f"[Source {index}]\n"
            f"File: {chunk['file_name']}\n"
            f"Page: {chunk['page']}\n"
            f"Text:\n{chunk['text']}"
            for index, chunk in enumerate(unique_chunks, start=1)
        )
        answer = llm_service.generate_answer(question=question, context=context)
        sources = [
            {
                "file_name": chunk["file_name"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
            }
            for chunk in unique_chunks
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(unique_chunks),
        }


rag_service = RAGService()
