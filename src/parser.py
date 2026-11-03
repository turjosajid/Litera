import re
from pathlib import Path
from typing import Optional
import pymupdf as fitz  # PyMuPDF

from src.models import PageContent, ParsedPaper


class PDFParser:
    """
    Extracts structured, page-tagged text from academic PDF documents using PyMuPDF.
    """

    def _clean_page_text(self, text: str) -> str:
        """
        Cleans hyphenated linebreaks and normalizes whitespace.
        Example: 'trans-\nformer' -> 'transformer'
        """
        if not text:
            return ""

        # 1. Rejoin words split with hyphens at line endings (e.g. 'atten-\ntion' -> 'attention')
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # 2. Replace multiple consecutive spaces and empty lines with clean spacing
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def parse_pdf(self, file_path: Path, paper_id: str) -> Optional[ParsedPaper]:
        """
        Opens a PDF file and extracts clean text page-by-page.
        Returns a ParsedPaper object with page numbers preserved.
        """
        if not file_path.exists():
            print(f"[Parser Error] File not found: {file_path}")
            return None

        try:
            doc = fitz.open(file_path)
            extracted_pages = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                raw_text = page.get_text("text")
                cleaned_text = self._clean_page_text(raw_text)

                if cleaned_text:
                    # Page numbers are 1-indexed (Page 1, Page 2, ...)
                    extracted_pages.append(
                        PageContent(page_number=page_idx + 1, text=cleaned_text)
                    )

            doc.close()

            return ParsedPaper(
                paper_id=paper_id,
                total_pages=len(extracted_pages),
                pages=extracted_pages
            )

        except Exception as e:
            print(f"[Parser Error] Failed to parse PDF {file_path}: {e}")
            return None