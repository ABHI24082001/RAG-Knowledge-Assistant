import re


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


class ChunkingService:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be 0 or greater.")
        if chunk_overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        chunks = []
        chunk_id = 1

        for page in pages:
            page_number = page["page"]
            text = re.sub(r"\s+", " ", page.get("text", "")).strip()

            if not text:
                continue

            start = 0
            text_length = len(text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)

                if end < text_length:
                    boundary_start = start + (self.chunk_size - self.chunk_overlap)
                    whitespace_boundary = text.rfind(" ", boundary_start, end)
                    sentence_boundary = text.rfind(". ", boundary_start, end)
                    boundary = max(whitespace_boundary, sentence_boundary + 1)

                    if boundary > start:
                        end = boundary

                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "page": page_number,
                            "text": chunk_text,
                            "characters": len(chunk_text),
                        }
                    )
                    chunk_id += 1

                if end >= text_length:
                    break

                start = max(end - self.chunk_overlap, start + 1)

        return chunks


chunking_service = ChunkingService()
