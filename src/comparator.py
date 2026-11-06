from typing import Optional
import ollama

from src.models import PaperAnalysis, PaperComparison


class PaperComparator:
    """
    Compares two research papers using structured synthesis and comparative reasoning.
    """

    def __init__(self, model_name: str = "qwen2.5:8b"):
        self.model_name = model_name

    def _format_paper_dossier(self, title: str, analysis: PaperAnalysis) -> str:
        """Formats a single paper's analysis into a concise, labeled dossier."""
        results_str = "\n".join(f"  • {r}" for r in analysis.key_results)
        limitations_str = "\n".join(f"  • {l}" for l in analysis.limitations)

        return f"""Title: {title} (ID: {analysis.paper_id})
TL;DR: {analysis.tldr}
Motivation: {analysis.motivation}
Proposed Methodology: {analysis.methodology}
Key Results:
{results_str}
Known Limitations:
{limitations_str}"""

    def compare_papers(
        self,
        paper_a_title: str,
        analysis_a: PaperAnalysis,
        paper_b_title: str,
        analysis_b: PaperAnalysis
    ) -> Optional[PaperComparison]:
        """
        Runs comparative synthesis between two papers and returns a structured PaperComparison.
        """
        dossier_a = self._format_paper_dossier(paper_a_title, analysis_a)
        dossier_b = self._format_paper_dossier(paper_b_title, analysis_b)

        system_prompt = (
            "You are a principal AI researcher conducting a rigorous comparative evaluation "
            "between two scientific papers. Compare their core architectures, benchmark results, "
            "computational efficiencies, and real-world tradeoffs. Be objective, precise, and practical."
        )

        user_prompt = f"""Please perform an in-depth comparative analysis of the following two research papers:

=== PAPER A ===
{dossier_a}

=== PAPER B ===
{dossier_b}

Generate a structured side-by-side comparison matrix and final recommendation."""

        print(f"[Comparing] Synthesizing '{paper_a_title}' vs '{paper_b_title}' with {self.model_name}...")

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format=PaperComparison.model_json_schema(),
                options={
                    "temperature": 0.2,  # Low temperature for objective comparative reasoning
                    "num_ctx": 8192
                }
            )

            response_text = response["message"]["content"]
            comparison = PaperComparison.model_validate_json(response_text)
            return comparison

        except Exception as e:
            print(f"[Comparison Error] Failed to compare papers: {e}")
            return None