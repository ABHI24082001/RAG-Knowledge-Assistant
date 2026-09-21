import pymupdf


class PDFService:
    def extract_text(self, pdf_bytes: bytes) -> dict:
        document = None

        try:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            pages = []

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                        "characters": len(text),
                    }
                )

            return {
                "page_count": len(pages),
                "total_characters": sum(page["characters"] for page in pages),
                "pages": pages,
            }
        finally:
            if document is not None:
                document.close()


pdf_service = PDFService()
