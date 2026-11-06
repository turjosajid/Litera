from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class PaperMetadata(BaseModel):
    """
    Represents standardized metadata for an academic research paper.
    This schema is used across search, storage, and retrieval.
    """
    paper_id: str = Field(
        ..., 
        description="The unique ArXiv ID (e.g., '1706.03762v7' or '2301.12345')"
    )
    title: str = Field(
        ..., 
        description="The title of the paper, stripped of unnecessary whitespace"
    )
    authors: List[str] = Field(
        default_factory=list, 
        description="List of author names"
    )
    abstract: str = Field(
        ..., 
        description="Clean abstract text without LaTeX/line break formatting artifacts"
    )
    published_date: datetime = Field(
        ..., 
        description="Date the paper was published or submitted"
    )
    pdf_url: str = Field(
        ..., 
        description="Direct download URL for the PDF document"
    )
    categories: List[str] = Field(
        default_factory=list, 
        description="ArXiv subject classifications (e.g., ['cs.AI', 'cs.CL'])"
    )
    primary_category: Optional[str] = Field(
        default=None, 
        description="The primary classification code of the paper"
    )

class PageContent(BaseModel):
    """Represents a single page extracted from a research paper."""
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Cleaned text content of this page")


class ParsedPaper(BaseModel):
    """Represents the complete parsed content of a paper."""
    paper_id: str
    total_pages: int
    pages: List[PageContent]

    @property
    def full_text(self) -> str:
        """Combines all pages into a single document string."""
        return "\n\n".join(f"--- Page {p.page_number} ---\n{p.text}" for p in self.pages)

class PaperAnalysis(BaseModel):
    """Structured deep analysis of a research paper produced by the LLM."""
    paper_id: str = Field(..., description="The ArXiv paper ID")
    tldr: str = Field(
        ..., 
        description="A concise 2-3 sentence executive summary of the paper"
    )
    motivation: str = Field(
        ..., 
        description="The core problem the authors are solving and why existing solutions were insufficient"
    )
    methodology: str = Field(
        ..., 
        description="The core technical approach, architecture, or algorithm proposed in the paper"
    )
    key_results: List[str] = Field(
        ..., 
        description="Key quantitative or qualitative benchmark results and findings"
    )
    limitations: List[str] = Field(
        ..., 
        description="Documented limitations, failure cases, or areas for future research"
    )

class PaperChunk(BaseModel):
    """Represents a discrete semantic chunk of a paper with spatial metadata."""
    chunk_id: str = Field(
        ..., 
        description="Unique ID"
    )
    paper_id: str = Field(..., description="The ArXiv paper ID")
    page_number: int = Field(..., description="1-indexed page number where this chunk lives")
    text: str = Field(..., description="The actual text content of this chunk")

class ComparisonPoint(BaseModel):
    """A single dimension of comparison between two papers."""
    aspect: str = Field(
        ..., 
        description="Dimension being compared (e.g. 'Core Architecture', 'Key Datasets', 'Computational Efficiency')"
    )
    paper_a_take: str = Field(..., description="How Paper A approaches this aspect")
    paper_b_take: str = Field(..., description="How Paper B approaches this aspect")
    winner_or_tradeoff: str = Field(
        ..., 
        description="Analysis of which approach performs better or the fundamental tradeoff between them"
    )


class PaperComparison(BaseModel):
    """Full structured comparative analysis between two research papers."""
    paper_a_title: str
    paper_b_title: str
    summary_of_differences: str = Field(
        ..., 
        description="High-level 2-3 sentence summary of how these two works diverge"
    )
    comparison_matrix: List[ComparisonPoint] = Field(
        ..., 
        description="List of specific criteria compared side-by-side"
    )
    verdict_and_recommendation: str = Field(
        ..., 
        description="Practical guidance on when to choose Paper A's methodology vs Paper B's methodology"
    )