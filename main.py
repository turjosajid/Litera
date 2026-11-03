import sys
from src.search import ArxivSearchEngine
from src.downloader import PDFDownloader
from src.parser import PDFParser
from src.analyzer import PaperAnalyzer
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
    """Prints the deep structured analysis in a clean technical layout."""
    print("\n" + "=" * 75)
    print(f" DEEP TECHNICAL ANALYSIS: {analysis.paper_id}")
    print("=" * 75)

    print("\n📌 TL;DR")
    print(f"   {analysis.tldr}")

    print("\n🎯 MOTIVATION & PROBLEM STATEMENT")
    print(f"   {analysis.motivation}")

    print("\n⚙️ PROPOSED METHODOLOGY & ARCHITECTURE")
    print(f"   {analysis.methodology}")

    print("\n📊 KEY EXPERIMENTAL RESULTS & BENCHMARKS")
    for result in analysis.key_results:
        print(f"   • {result}")

    print("\n⚠️ LIMITATIONS & FUTURE WORK")
    for lim in analysis.limitations:
        print(f"   • {lim}")

    print("\n" + "=" * 75 + "\n")


def main():
    # Initialize our pipeline services
    search_engine = ArxivSearchEngine()
    downloader = PDFDownloader()
    parser = PDFParser()
    # Note: if your Ollama model tag differs, change it here (e.g. "qwen2.5:8b")
    analyzer = PaperAnalyzer(model_name="qwen3:8b")

    print("=" * 75)
    print(" Research Paper Intelligence - End-to-End Analysis CLI")
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

            print(f"\nFound {len(papers)} papers:")
            print("-" * 75)
            for idx, p in enumerate(papers, start=1):
                display_paper_card(idx, p)

            # Let the user choose a paper to analyze
            choice = input("\nEnter paper number to analyze (or press Enter to search again): ").strip()
            if not choice.isdigit() or not (1 <= int(choice) <= len(papers)):
                continue

            selected_paper = papers[int(choice) - 1]
            print(f"\nSelected: '{selected_paper.title}'")

            # 1. Download & Cache PDF
            print("\n[Step 1/3] Checking cache / downloading PDF...")
            pdf_path = downloader.download_paper(
                paper_id=selected_paper.paper_id,
                pdf_url=selected_paper.pdf_url
            )
            if not pdf_path:
                print("[Error] Could not download or access PDF. Skipping.")
                continue

            # 2. Parse Layout & Pages
            print("\n[Step 2/3] Parsing PDF pages...")
            parsed_paper = parser.parse_pdf(pdf_path, paper_id=selected_paper.paper_id)
            if not parsed_paper or parsed_paper.total_pages == 0:
                print("[Error] Could not extract text from PDF. Skipping.")
                continue
            print(f"[Parsed] Successfully extracted {parsed_paper.total_pages} pages.")

            # 3. Analyze with Ollama
            print("\n[Step 3/3] Generating deep analysis with Ollama...")
            analysis = analyzer.analyze_paper(parsed_paper)

            if analysis:
                display_analysis_dossier(analysis)
            else:
                print("[Error] Analysis failed. Please check Ollama status.")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()