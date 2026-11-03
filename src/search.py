import re
from typing import List, Optional
import arxiv

from src.models import PaperMetadata


class ArxivSearchEngine:
    """
    Search engine client for fetching and normalizing research papers from ArXiv.
    """

    def __init__(self, page_size: int = 10, delay_seconds: float = 3.0, num_retries: int = 3):
        # ArXiv recommends at least 3 seconds delay between batch queries
        self.client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries
        )

    def _clean_text(self, text: Optional[str]) -> str:
        """
        Cleans LaTeX linebreaks, tabs, and excess whitespace.
        """
        if not text:
            return ""
        # Replace newlines and tabs with single spaces
        cleaned = re.sub(r"[\n\r\t]+", " ", text)
        # Collapse multiple spaces into one and strip edges
        return re.sub(r"\s+", " ", cleaned).strip()

    def search(
        self, 
        query: str, 
        max_results: int = 5, 
        sort_by: str = "relevance"
    ) -> List[PaperMetadata]:
        """
        Searches ArXiv for papers matching the query and returns clean PaperMetadata objects.

        :param query: The search term (e.g. 'diffusion models' or 'au:bengio')
        :param max_results: Number of papers to fetch
        :param sort_by: 'relevance' or 'submitted_date'
        :return: List of validated PaperMetadata objects
        """
        if not query or not query.strip():
            return []

        # Map string sort choices to ArXiv sort criteria
        criterion = (
            arxiv.SortCriterion.SubmittedDate 
            if sort_by == "submitted_date" 
            else arxiv.SortCriterion.Relevance
        )

        search_query = arxiv.Search(
            query=query.strip(),
            max_results=max_results,
            sort_by=criterion,
            sort_order=arxiv.SortOrder.Descending
        )

        papers: List[PaperMetadata] = []

        try:
            for result in self.client.results(search_query):
                paper = PaperMetadata(
                    paper_id=result.get_short_id(),
                    title=self._clean_text(result.title),
                    authors=[author.name for author in result.authors],
                    abstract=self._clean_text(result.summary),
                    published_date=result.published,
                    pdf_url=result.pdf_url,
                    categories=result.categories,
                    primary_category=result.primary_category
                )
                papers.append(paper)
        except Exception as e:
            print(f"[Error] Failed to fetch or parse papers from ArXiv: {e}")
            return []

        return papers


