from typing import Dict, Any, List
import ollama

from src.models import PaperChunk
from src.vector_store import VectorStoreManager


class GroundedQAEngine:
    """
    Orchestrates semantic retrieval and generates citation-backed answers
    grounded strictly in the paper's text.
    """

    def __init__(
        self, 
        vector_store: VectorStoreManager, 
        model_name: str = "qwen2.5:8b"
    ):
        self.vector_store = vector_store
        self.model_name = model_name

    def _format_context(self, chunks: List[PaperChunk]) -> str:
        """
        Formats retrieved chunks into clear, tagged source excerpts.
        """
        context_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            block = (
                f"[SOURCE {idx} | Page {chunk.page_number} | ID: {chunk.chunk_id}]\n"
                f"{chunk.text}"
            )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def answer_question(
        self, 
        query: str, 
        paper_id: str, 
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Retrieves relevant chunks and generates a grounded answer with page citations.
        """
        # 1. Semantic Retrieval from ChromaDB
        chunks = self.vector_store.search_relevant_chunks(
            query=query, 
            paper_id=paper_id, 
            top_k=top_k
        )

        if not chunks:
            return {
                "answer": "No relevant sections found in this paper for your query.",
                "sources": []
            }

        context = self._format_context(chunks)

        # 2. Defensive Grounding Prompt
        system_prompt = (
            "You are a rigorous academic AI research assistant. "
            "Your task is to answer questions about a research paper using ONLY the provided excerpts.\n\n"
            "STRICT RULES:\n"
            "1. Base your answer strictly on the provided excerpts. Do NOT bring in external knowledge.\n"
            "2. For every factual claim, cite the exact page number in brackets, for example: [Page 4].\n"
            "3. If different parts of your answer come from different pages, cite each one accordingly.\n"
            "4. If the provided excerpts do not contain enough information to answer, state clearly: "
            "'This specific detail is not mentioned in the retrieved sections of the paper.'\n"
            "5. Never fabricate, extrapolate, or hallucinate citations."
        )

        user_prompt = f"""Excerpts from the paper:
----------------------------------------
{context}
----------------------------------------

Question: {query}

Please provide a clear, accurate answer with [Page X] citations:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.1,  # Strict factual mode
                    "num_ctx": 8192
                }
            )

            answer = response["message"]["content"]

            # Extract unique pages used as sources
            cited_pages = sorted(list(set(chunk.page_number for chunk in chunks)))

            return {
                "answer": answer.strip(),
                "sources": chunks,
                "pages": cited_pages
            }

        except Exception as e:
            return {
                "answer": f"[Error generating answer]: {e}",
                "sources": [],
                "pages": []
            }