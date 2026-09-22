import logging
import threading
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self._load_lock = threading.Lock()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            # Defer Torch/model import and allocation until this service is used.
            from sentence_transformers import SentenceTransformer

            settings = get_settings()
            logger.info("Loading embedding model %s on %s", settings.embedding_model_name, settings.embedding_device)
            self.model = SentenceTransformer(settings.embedding_model_name, device=settings.embedding_device)
            logger.info("Embedding model loaded successfully")

    def generate(self, text: str) -> list[float]:
        self._load_model()
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def generate_many(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=get_settings().embedding_batch_size,
        )
        return embeddings.tolist()


embedding_service = EmbeddingService()
