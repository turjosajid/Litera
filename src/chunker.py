from typing import List
from src.models import ParsedPaper, PageContent, PaperChunk


class TextChunker:
    """
    Splits parsed paper pages into semantic, paragraph-aware chunks
    while preserving page numbers and attaching unique metadata.
    """

    def __init__(self, target_chunk_size: int = 800, overlap: int = 150):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def _chunk_page_text(self, page: PageContent, paper_id: str) -> List[PaperChunk]:
        """
        Splits a single page into overlapping chunks respecting paragraph boundaries.
        """
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        chunks: List[PaperChunk] = []

        current_text = ""
        chunk_idx = 1

        for para in paragraphs:
            # If adding this paragraph exceeds our target size and we already have content:
            if len(current_text) + len(para) > self.target_chunk_size and current_text:
                # Save the current chunk
                chunk_id = f"{paper_id}_p{page.page_number}_c{chunk_idx}"
                chunks.append(
                    PaperChunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        page_number=page.page_number,
                        text=current_text.strip()
                    )
                )
                chunk_idx += 1

                # Start the next chunk with an overlap from the end of current text
                overlap_text = current_text[-self.overlap:] if len(current_text) > self.overlap else ""
                current_text = overlap_text + " " + para
            else:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para

        # Don't forget any remaining text on the page
        if current_text.strip():
            chunk_id = f"{paper_id}_p{page.page_number}_c{chunk_idx}"
            chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    page_number=page.page_number,
                    text=current_text.strip()
                )
            )

        return chunks

    def chunk_paper(self, parsed_paper: ParsedPaper) -> List[PaperChunk]:
        """
        Processes all pages in a paper and returns a list of indexed PaperChunks.
        """
        all_chunks: List[PaperChunk] = []

        for page in parsed_paper.pages:
            page_chunks = self._chunk_page_text(page, paper_id=parsed_paper.paper_id)
            all_chunks.extend(page_chunks)

        print(f"[Chunking] Generated {len(all_chunks)} semantic chunks across {parsed_paper.total_pages} pages.")
        return all_chunks