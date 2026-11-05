import sys
from src.search import ArxivSearchEngine
from src.downloader import PDFDownloader
from src.parser import PDFParser
from src.analyzer import PaperAnalyzer
from src.chunker import TextChunker
from src.vector_store import VectorStoreManager
from src.qa_engine import GroundedQAEngine
from src.models import PaperAnalysis


def display_paper_card(index: int, paper):
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
    chunker = TextChunker()
    vector_store = VectorStoreManager(embedding_model="nomic-embed-text")
    qa_engine = GroundedQAEngine(vector_store=vector_store, model_name="qwen3:8b")

    print("=" * 75)
    print(" Research Paper Intelligence - Full RAG & Analysis System")
    print("=" * 75)

    while True:
        try:
            query = input("\nEnter search query (or 'exit' to quit): ").strip()
            if not query or query.lower() in ("exit", "quit", "q"):
                print("Exiting. Happy researching!")
                break

            print(f"\nSearching ArXiv for: '{query}'...")
            papers = search_engine.search(query=query, max_results=3)

            if not papers:
                print("No papers found. Try different keywords.")
                continue

            for idx, p in enumerate(papers, start=1):
                display_paper_card(idx, p)

            choice = input("\nEnter paper number to explore (or Enter to search again): ").strip()
            if not choice.isdigit() or not (1 <= int(choice) <= len(papers)):
                continue

            selected_paper = papers[int(choice) - 1]
            paper_id = selected_paper.paper_id
            print(f"\nSelected: '{selected_paper.title}'")

            # 1. Download & Cache PDF
            print("\n[Phase 1] Checking cache / downloading PDF...")
            pdf_path = downloader.download_paper(paper_id=paper_id, pdf_url=selected_paper.pdf_url)
            if not pdf_path:
                continue

            # 2. Parse Layout & Pages
            print("\n[Phase 2] Parsing PDF pages...")
            parsed_paper = parser.parse_pdf(pdf_path, paper_id=paper_id)
            if not parsed_paper or parsed_paper.total_pages == 0:
                continue

            # 3. Analyze Paper
            print("\n[Phase 3] Generating deep analysis...")
            analysis = analyzer.analyze_paper(parsed_paper)
            if analysis:
                display_analysis_dossier(analysis)

            # 4. Chunk & Index for Grounded RAG
            print("\n[Phase 4] Indexing paper into ChromaDB for Q&A...")
            chunks = chunker.chunk_paper(parsed_paper)
            vector_store.index_chunks(chunks)

            # 5. Start Interactive Q&A Session
            run_qa_session(qa_engine, paper_id=paper_id, paper_title=selected_paper.title)

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()