import sys
from typing import Dict, Optional
from src.search import ArxivSearchEngine
from src.downloader import PDFDownloader
from src.parser import PDFParser
from src.analyzer import PaperAnalyzer
from src.chunker import TextChunker
from src.vector_store import VectorStoreManager
from src.qa_engine import GroundedQAEngine
from src.comparator import PaperComparator
from src.models import PaperMetadata, PaperAnalysis, PaperComparison


# In-memory session cache so we never re-analyze the same paper twice in one session
ANALYSIS_CACHE: Dict[str, PaperAnalysis] = {}


def display_paper_card(index: int, paper: PaperMetadata):
    """Prints paper summary card in search results."""
    print(f"\n[{index}] {paper.title}")
    print(f"    ID:        {paper.paper_id}")
    print(f"    Authors:   {', '.join(paper.authors[:3])}{' et al.' if len(paper.authors) > 3 else ''}")
    print(f"    Published: {paper.published_date.strftime('%Y-%m-%d')}")
    print(f"    Category:  {paper.primary_category}")
    abstract_preview = (paper.abstract[:180] + "...") if len(paper.abstract) > 180 else paper.abstract
    print(f"    Abstract:  {abstract_preview}")
    print("-" * 75)


def display_analysis_dossier(analysis: PaperAnalysis):
    """Prints the deep structured analysis."""
    print("\n" + "=" * 75)
    print(f" DEEP TECHNICAL ANALYSIS: {analysis.paper_id}")
    print("=" * 75)
    print(f"\n📌 TL;DR:\n   {analysis.tldr}")
    print(f"\n🎯 MOTIVATION & PROBLEM STATEMENT:\n   {analysis.motivation}")
    print(f"\n⚙️ PROPOSED METHODOLOGY:\n   {analysis.methodology}")
    print("\n📊 KEY EXPERIMENTAL RESULTS:")
    for result in analysis.key_results:
        print(f"   • {result}")
    print("\n⚠️ LIMITATIONS & FUTURE WORK:")
    for lim in analysis.limitations:
        print(f"   • {lim}")
    print("=" * 75)


def display_comparison_dossier(comparison: PaperComparison):
    """Prints the side-by-side comparison matrix and verdict."""
    print("\n" + "=" * 75)
    print(" ⚖️ SIDE-BY-SIDE RESEARCH PAPER COMPARISON")
    print(f" Paper A: {comparison.paper_a_title}")
    print(f" Paper B: {comparison.paper_b_title}")
    print("=" * 75)

    print("\n📌 EXECUTIVE SUMMARY OF DIVERGENCE:")
    print(f"   {comparison.summary_of_differences}")

    print("\n📊 COMPARISON MATRIX:")
    for idx, point in enumerate(comparison.comparison_matrix, start=1):
        print(f"\n   [{idx}] Dimension: {point.aspect.upper()}")
        print(f"       • Paper A: {point.paper_a_take}")
        print(f"       • Paper B: {point.paper_b_take}")
        print(f"       ⚖️  Tradeoff / Winner: {point.winner_or_tradeoff}")

    print("\n" + "-" * 75)
    print("🏆 FINAL VERDICT & RECOMMENDATION:")
    print(f"   {comparison.verdict_and_recommendation}")
    print("=" * 75 + "\n")


def get_paper_analysis(
    paper: PaperMetadata,
    downloader: PDFDownloader,
    parser: PDFParser,
    analyzer: PaperAnalyzer
) -> Optional[PaperAnalysis]:
    """Helper to fetch, parse, and analyze a paper with session caching."""
    if paper.paper_id in ANALYSIS_CACHE:
        print(f"[Session Cache Hit] Reusing existing analysis for: {paper.title}")
        return ANALYSIS_CACHE[paper.paper_id]

    print(f"\n[Processing] '{paper.title}'...")
    pdf_path = downloader.download_paper(paper_id=paper.paper_id, pdf_url=paper.pdf_url)
    if not pdf_path:
        return None

    parsed_paper = parser.parse_pdf(pdf_path, paper_id=paper.paper_id)
    if not parsed_paper or parsed_paper.total_pages == 0:
        return None

    analysis = analyzer.analyze_paper(parsed_paper)
    if analysis:
        ANALYSIS_CACHE[paper.paper_id] = analysis

    return analysis


def run_qa_session(qa_engine: GroundedQAEngine, paper_id: str, paper_title: str):
    """Interactive loop allowing the user to ask grounded questions with citations."""
    print("\n" + "=" * 75)
    print(f" 💬 INTERACTIVE GROUNDED Q&A: '{paper_title}'")
    print(" Ask any question about the paper. The AI will cite exact page numbers.")
    print(" Type 'back' to return to search, or 'exit' to quit.")
    print("=" * 75)

    while True:
        try:
            query = input("\n❓ Ask question: ").strip()
            if not query:
                continue
            if query.lower() in ("back", "b"):
                break
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting application. Goodbye!")
                sys.exit(0)

            print("[Retrieving & Synthesizing] Searching vectors and consulting paper text...")
            result = qa_engine.answer_question(query=query, paper_id=paper_id, top_k=4)

            print("\n💡 ANSWER:")
            print(result["answer"])

            if result["pages"]:
                pages_str = ", ".join(f"Page {p}" for p in result["pages"])
                print(f"\n📑 Sources Consulted: {pages_str} ({len(result['sources'])} excerpts retrieved)")
            print("-" * 75)

        except KeyboardInterrupt:
            break


def main():
    # Initialize all modular services
    search_engine = ArxivSearchEngine()
    downloader = PDFDownloader()
    parser = PDFParser()
    analyzer = PaperAnalyzer(model_name="qwen3:8b", num_ctx=16384)
    comparator = PaperComparator(model_name="qwen3:8b")
    chunker = TextChunker()
    vector_store = VectorStoreManager(embedding_model="nomic-embed-text")
    qa_engine = GroundedQAEngine(vector_store=vector_store, model_name="qwen3:8b")

    print("=" * 75)
    print(" Research Paper Intelligence - Full RAG, Analysis & Comparison")
    print("=" * 75)

    while True:
        try:
            query = input("\nEnter search query (or 'exit' to quit): ").strip()
            if not query or query.lower() in ("exit", "quit", "q"):
                print("Exiting. Happy researching!")
                break

            print(f"\nSearching ArXiv for: '{query}'...")
            papers = search_engine.search(query=query, max_results=4)

            if not papers:
                print("No papers found. Try different keywords.")
                continue

            for idx, p in enumerate(papers, start=1):
                display_paper_card(idx, p)

            user_choice = input(
                "\n👉 Select an option:\n"
                "   • Enter paper number (e.g. '1') to analyze & chat\n"
                "   • Type 'compare X Y' (e.g. 'compare 1 2') to compare two papers\n"
                "   • Press Enter to search again\n"
                "Choice: "
            ).strip()

            if not user_choice:
                continue

            # MODE A: MULTI-PAPER COMPARISON ('compare X Y')
            if user_choice.lower().startswith("compare"):
                parts = user_choice.split()
                if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
                    print("[Error] Please specify two valid numbers, e.g. 'compare 1 2'")
                    continue

                idx_a, idx_b = int(parts[1]) - 1, int(parts[2]) - 1
                if not (0 <= idx_a < len(papers) and 0 <= idx_b < len(papers)) or idx_a == idx_b:
                    print("[Error] Invalid paper indices selected for comparison.")
                    continue

                paper_a, paper_b = papers[idx_a], papers[idx_b]

                analysis_a = get_paper_analysis(paper_a, downloader, parser, analyzer)
                analysis_b = get_paper_analysis(paper_b, downloader, parser, analyzer)

                if not analysis_a or not analysis_b:
                    print("[Error] Could not extract analysis for both papers.")
                    continue

                comparison = comparator.compare_papers(
                    paper_a_title=paper_a.title,
                    analysis_a=analysis_a,
                    paper_b_title=paper_b.title,
                    analysis_b=analysis_b
                )

                if comparison:
                    display_comparison_dossier(comparison)

            # MODE B: SINGLE PAPER EXPLORATION & Q&A
            elif user_choice.isdigit() and 1 <= int(user_choice) <= len(papers):
                selected_paper = papers[int(user_choice) - 1]
                paper_id = selected_paper.paper_id
                print(f"\nSelected: '{selected_paper.title}'")

                analysis = get_paper_analysis(selected_paper, downloader, parser, analyzer)
                if analysis:
                    display_analysis_dossier(analysis)

                print("\n[Indexing] Ensuring paper is indexed in ChromaDB for Q&A...")
                if not vector_store.is_paper_indexed(paper_id):
                    pdf_path = downloader.get_local_path(paper_id)
                    parsed_paper = parser.parse_pdf(pdf_path, paper_id=paper_id)
                    if parsed_paper:
                        chunks = chunker.chunk_paper(parsed_paper)
                        vector_store.index_chunks(chunks)

                run_qa_session(qa_engine, paper_id=paper_id, paper_title=selected_paper.title)

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()