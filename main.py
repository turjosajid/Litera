import sys
from typing import Dict, Optional, List
from src.search import ArxivSearchEngine
from src.downloader import PDFDownloader
from src.parser import PDFParser
from src.analyzer import PaperAnalyzer
from src.chunker import TextChunker
from src.vector_store import VectorStoreManager
from src.qa_engine import GroundedQAEngine
from src.comparator import PaperComparator
from src.database import DatabaseManager
from src.models import PaperMetadata, PaperAnalysis, PaperComparison

# In-memory session cache so we never re-analyze the same paper twice in one session
ANALYSIS_CACHE: Dict[str, PaperAnalysis] = {}


def display_paper_card(index: int, paper: PaperMetadata):
    """Prints paper summary card in search/folder results."""
    print(f"\n[{index}] {paper.title}")
    print(f"    ID:        {paper.paper_id}")
    authors_str = ', '.join(paper.authors[:3]) + (' et al.' if len(paper.authors) > 3 else '')
    print(f"    Authors:   {authors_str}")
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
    print(" Type 'back' to return to menu, or 'exit' to quit.")
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


def save_paper_flow(paper: PaperMetadata, db: DatabaseManager):
    """Guides user through saving a paper to an existing or new folder."""
    folders = db.list_folders()

    print(f"\n📁 Select a folder to bookmark: '{paper.title}'")
    if folders:
        for f in folders:
            print(f"   [{f['id']}] {f['name']} ({f['paper_count']} papers)")
    else:
        print("   (No folders created yet!)")

    choice = input("\nEnter Folder ID (or type 'new <folder_name>' to create one): ").strip()

    if choice.lower().startswith("new "):
        new_name = choice[4:].strip()
        folder_id = db.create_folder(new_name)
        if not folder_id:
            return
        print(f"[Success] Created folder '{new_name}' (ID: {folder_id})")
    elif choice.isdigit():
        folder_id = int(choice)
    else:
        print("[Notice] Save canceled.")
        return

    if db.save_paper_to_folder(paper, folder_id):
        print(f"⭐ [Saved] Successfully bookmarked '{paper.title}' into folder {folder_id}!")


def explore_single_paper(
    paper: PaperMetadata,
    downloader: PDFDownloader,
    parser: PDFParser,
    analyzer: PaperAnalyzer,
    chunker: TextChunker,
    vector_store: VectorStoreManager,
    qa_engine: GroundedQAEngine
):
    """Executes full analysis and enters Q&A session for a selected paper."""
    paper_id = paper.paper_id
    print(f"\nSelected: '{paper.title}'")

    analysis = get_paper_analysis(paper, downloader, parser, analyzer)
    if analysis:
        display_analysis_dossier(analysis)

    print("\n[Indexing] Ensuring paper is indexed in ChromaDB for Q&A...")
    if not vector_store.is_paper_indexed(paper_id):
        pdf_path = downloader.get_local_path(paper_id)
        parsed_paper = parser.parse_pdf(pdf_path, paper_id=paper_id)
        if parsed_paper:
            chunks = chunker.chunk_paper(parsed_paper)
            vector_store.index_chunks(chunks)

    run_qa_session(qa_engine, paper_id=paper_id, paper_title=paper.title)


def library_menu_flow(
    db: DatabaseManager,
    downloader: PDFDownloader,
    parser: PDFParser,
    analyzer: PaperAnalyzer,
    chunker: TextChunker,
    vector_store: VectorStoreManager,
    qa_engine: GroundedQAEngine,
    comparator: PaperComparator
):
    """Sub-menu for viewing, managing, and launching analysis from saved folders."""
    while True:
        print("\n" + "=" * 75)
        print(" 📁 MY RESEARCH LIBRARY & FOLDERS")
        print("=" * 75)
        folders = db.list_folders()

        if not folders:
            print("You have no saved folders yet.")
        else:
            for f in folders:
                print(f"  [{f['id']}] 📂 {f['name']} ({f['paper_count']} papers)")

        print("\nOptions:")
        print("  • Enter Folder ID to open that folder")
        print("  • Type 'new <name>' to create a new folder")
        print("  • Type 'del <folder_id>' to delete a folder")
        print("  • Type 'back' to return to Main Menu")

        choice = input("\nChoice: ").strip()
        if choice.lower() in ("back", "b"):
            break

        if choice.lower().startswith("new "):
            name = choice[4:].strip()
            new_id = db.create_folder(name)
            if new_id:
                print(f"[Success] Created folder '{name}'!")
            continue

        if choice.lower().startswith("del "):
            del_id = choice[4:].strip()
            if del_id.isdigit():
                if db.delete_folder(int(del_id)):
                    print(f"[Success] Deleted folder {del_id}.")
            continue

        if choice.isdigit():
            folder_id = int(choice)
            folder_papers = db.get_papers_in_folder(folder_id)

            if not folder_papers:
                print(f"\nFolder {folder_id} is currently empty. Bookmark papers into it from ArXiv search!")
                continue

            print(f"\n--- Papers in Folder {folder_id} ---")
            for idx, p in enumerate(folder_papers, start=1):
                display_paper_card(idx, p)

            action = input(
                "\n👉 Select action:\n"
                "   • Enter paper # to Read Analysis & Chat\n"
                "   • Type 'compare X Y' to compare two papers from this folder\n"
                "   • Type 'remove X' to remove a paper from this folder\n"
                "   • Press Enter to go back to folders\n"
                "Choice: "
            ).strip()

            if not action:
                continue

            if action.isdigit() and 1 <= int(action) <= len(folder_papers):
                explore_single_paper(
                    folder_papers[int(action) - 1],
                    downloader, parser, analyzer, chunker, vector_store, qa_engine
                )
            elif action.lower().startswith("remove "):
                parts = action.split()
                if len(parts) == 2 and parts[1].isdigit():
                    rem_idx = int(parts[1]) - 1
                    if 0 <= rem_idx < len(folder_papers):
                        target_paper = folder_papers[rem_idx]
                        db.remove_paper_from_folder(folder_id, target_paper.paper_id)
                        print(f"[Removed] '{target_paper.title}' removed from folder {folder_id}.")
            elif action.lower().startswith("compare"):
                parts = action.split()
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                    idx_a, idx_b = int(parts[1]) - 1, int(parts[2]) - 1
                    if 0 <= idx_a < len(folder_papers) and 0 <= idx_b < len(folder_papers) and idx_a != idx_b:
                        p_a, p_b = folder_papers[idx_a], folder_papers[idx_b]
                        a_a = get_paper_analysis(p_a, downloader, parser, analyzer)
                        a_b = get_paper_analysis(p_b, downloader, parser, analyzer)
                        if a_a and a_b:
                            comp = comparator.compare_papers(p_a.title, a_a, p_b.title, a_b)
                            if comp:
                                display_comparison_dossier(comp)


def main():
    # Initialize all modular services
    search_engine = ArxivSearchEngine()
    downloader = PDFDownloader()
    parser = PDFParser()
    analyzer = PaperAnalyzer(model_name="qwen2.5:8b", num_ctx=16384)
    comparator = PaperComparator(model_name="qwen2.5:8b")
    chunker = TextChunker()
    vector_store = VectorStoreManager(embedding_model="nomic-embed-text")
    qa_engine = GroundedQAEngine(vector_store=vector_store, model_name="qwen2.5:8b")
    db = DatabaseManager()

    print("=" * 75)
    print(" 🔬 RESEARCH PAPER INTELLIGENCE - FULL PLATFORM")
    print("=" * 75)

    while True:
        try:
            print("\nMAIN MENU:")
            print("  [1] 🔍 Search ArXiv for Papers")
            print("  [2] 📁 My Research Library & Folders")
            print("  [3] ❌ Exit")

            menu_choice = input("\nEnter choice (1-3): ").strip()

            if menu_choice == "3" or menu_choice.lower() in ("exit", "quit", "q"):
                print("Thank you for using Research Paper Intelligence. Goodbye!")
                break

            elif menu_choice == "2":
                library_menu_flow(
                    db, downloader, parser, analyzer, chunker, vector_store, qa_engine, comparator
                )

            elif menu_choice == "1":
                query = input("\nEnter search query (or 'back' to menu): ").strip()
                if not query or query.lower() in ("back", "b"):
                    continue

                print(f"\nSearching ArXiv for: '{query}'...")
                papers = search_engine.search(query=query, max_results=4)

                if not papers:
                    print("No papers found. Try different keywords.")
                    continue

                for idx, p in enumerate(papers, start=1):
                    display_paper_card(idx, p)

                user_choice = input(
                    "\n👉 Select an option:\n"
                    "   • Enter paper # (e.g. '1') to analyze & chat\n"
                    "   • Type 'compare X Y' (e.g. 'compare 1 2') to compare two papers\n"
                    "   • Type 'save X' (e.g. 'save 1') to bookmark a paper into a folder\n"
                    "   • Press Enter to search again\n"
                    "Choice: "
                ).strip()

                if not user_choice:
                    continue

                if user_choice.lower().startswith("save "):
                    parts = user_choice.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        save_idx = int(parts[1]) - 1
                        if 0 <= save_idx < len(papers):
                            save_paper_flow(papers[save_idx], db)
                    continue

                if user_choice.lower().startswith("compare"):
                    parts = user_choice.split()
                    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                        idx_a, idx_b = int(parts[1]) - 1, int(parts[2]) - 1
                        if 0 <= idx_a < len(papers) and 0 <= idx_b < len(papers) and idx_a != idx_b:
                            paper_a, paper_b = papers[idx_a], papers[idx_b]
                            analysis_a = get_paper_analysis(paper_a, downloader, parser, analyzer)
                            analysis_b = get_paper_analysis(paper_b, downloader, parser, analyzer)
                            if analysis_a and analysis_b:
                                comparison = comparator.compare_papers(
                                    paper_a.title, analysis_a, paper_b.title, analysis_b
                                )
                                if comparison:
                                    display_comparison_dossier(comparison)
                    continue

                if user_choice.isdigit() and 1 <= int(user_choice) <= len(papers):
                    explore_single_paper(
                        papers[int(user_choice) - 1],
                        downloader, parser, analyzer, chunker, vector_store, qa_engine
                    )

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()