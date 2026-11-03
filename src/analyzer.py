import re
from typing import Optional
import ollama
from src.models import ParsedPaper, PaperAnalysis


class PaperAnalyzer:
    """
    Orchestrates paper summarization and technical analysis using a local Ollama model.
    """

    def __init__(self, model_name: str = "qwen3:8b"):  
        self.model_name = model_name

    def _prepare_context(self, parsed_paper: ParsedPaper, max_pages: int = 12) -> str:
        """
        Intelligently extracts the core paper content:
        - Includes up to 'max_pages'
        - Cuts off automatically when 'References' or 'Bibliography' is reached
        """
        context_parts = []
        
        # Common headers that signal the end of the main paper text
        ref_patterns = [r"\nreferences\b", r"\nbibliography\b", r"\n\d+\s+references\b"]
        for page in parsed_paper.pages[:max_pages]:
            text_lower = page.text.lower()
            
            # Check if the references section starts on this page
            cut_index = -1
            for pattern in ref_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    cut_index = match.start()
                    break
            if cut_index != -1:
                # Add text up to the start of references, then stop
                clean_text_before_refs = page.text[:cut_index].strip()
                if clean_text_before_refs:
                    context_parts.append(f"--- PAGE {page.page_number} ---\n{clean_text_before_refs}")
                break
            else:
                context_parts.append(f"--- PAGE {page.page_number} ---\n{page.text}")
        return "\n\n".join(context_parts)

    def analyze_paper(self, parsed_paper: ParsedPaper) -> Optional[PaperAnalysis]:
        context = self._prepare_context(parsed_paper, max_pages=8)

        system_prompt = (
            "You are an expert AI and Machine Learning research scientist. "
            "Your task is to analyze the provided research paper thoroughly and extract "
            "the core scientific insights. Be precise, technical, and objective. "
            "IMPORTANT: Only extract facts present in the text. If a specific detail "
            "(such as ablation studies or limitations) is omitted from the excerpt, "
            "explicitly state 'Not present in provided excerpt' rather than guessing."
        )

        user_prompt = f"""Please analyze the following academic paper text and return a JSON object.

Paper ID: {parsed_paper.paper_id}
Total Pages: {parsed_paper.total_pages}

Document Content:
{context}
"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format=PaperAnalysis.model_json_schema(),
                options={
                    "temperature": 0.2,
                    "num_ctx": 16384  # Crucial: prevents Ollama from silently truncating past 2048/4096 tokens
                }
            )

            response_text = response["message"]["content"]
            return PaperAnalysis.model_validate_json(response_text)

        except Exception as e:
            print(f"[Analysis Error] Failed to analyze paper with Ollama: {e}")
            return None