from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from app.services.llm_service import NO_ANSWER
from app.services.rag_service import RAGService
from app.services.vector_service import VECTOR_SIZE


def chunk(
    score: float,
    *,
    document_id: str = "document-a",
    chunk_id: int = 1,
    text: str = "Abhishek Kumar is a React Native developer with 2.5 years of experience.",
) -> dict:
    return {
        "id": f"{document_id}-{chunk_id}",
        "score": score,
        "text": text,
        "source_type": "document",
        "document_id": document_id,
        "file_name": f"{document_id}.pdf",
        "page": 1,
        "chunk_id": chunk_id,
    }


class RAGServiceTests(TestCase):
    def setUp(self) -> None:
        self.embedding = MagicMock()
        self.embedding.generate.return_value = [0.0] * VECTOR_SIZE
        self.vector = MagicMock()
        self.llm = MagicMock()
        self.llm.generate_answer.return_value = "Abhishek Kumar has 2.5 years of experience."
        self.settings = SimpleNamespace(rag_min_relevance_score=0.20)
        self.patches = [
            patch("app.services.rag_service.embedding_service", self.embedding),
            patch("app.services.rag_service.vector_service", self.vector),
            patch("app.services.rag_service.llm_service", self.llm),
            patch("app.services.rag_service.get_settings", return_value=self.settings),
        ]
        for active_patch in self.patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)
        self.service = RAGService()

    def test_relevant_question_calls_llm_and_returns_sources(self) -> None:
        self.vector.search_documents.return_value = [chunk(0.33), chunk(0.15, chunk_id=2)]

        result = self.service.answer_question("How much experience does Abhishek Kumar have?")

        self.llm.generate_answer.assert_called_once()
        self.assertEqual(result["retrieved_chunks"], 1)
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("2.5 years", result["answer"])

    def test_unrelated_question_does_not_call_llm(self) -> None:
        self.vector.search_documents.return_value = [chunk(0.09), chunk(0.03, chunk_id=2)]

        result = self.service.answer_question("What is the capital of Mars?")

        self.llm.generate_answer.assert_not_called()
        self.assertEqual(result["answer"], NO_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks"], 0)

    def test_selected_document_isolation_is_preserved(self) -> None:
        self.vector.search_documents.return_value = [chunk(0.10, document_id="document-b")]

        result = self.service.answer_question("Question only answered by document A", document_id="document-b")

        self.vector.search_documents.assert_called_once_with(
            vector=[0.0] * VECTOR_SIZE,
            limit=3,
            document_id="document-b",
        )
        self.llm.generate_answer.assert_not_called()
        self.assertEqual(result["answer"], NO_ANSWER)

    def test_no_documents_does_not_call_llm(self) -> None:
        self.vector.search_documents.return_value = []

        result = self.service.answer_question("Any question")

        self.llm.generate_answer.assert_not_called()
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks"], 0)

    def test_duplicate_document_chunks_are_removed(self) -> None:
        duplicate = chunk(0.40)
        self.vector.search_documents.return_value = [duplicate, {**duplicate, "id": "another-point"}]

        result = self.service.answer_question("How much experience does Abhishek Kumar have?")

        self.llm.generate_answer.assert_called_once()
        self.assertEqual(result["retrieved_chunks"], 1)
        self.assertEqual(len(result["sources"]), 1)
