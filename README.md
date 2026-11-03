# Litera: Research Paper Intelligence 

An AI-powered research assistant that searches, downloads, parses, and performs deep technical analysis on academic papers from ArXiv using local LLMs (Ollama).

## Features 
- **ArXiv Paper Search:** Rate-limited and validated search engine retrieving paper metadata and direct PDF URLs.
- **Local PDF Caching:** Automated stream-downloading and disk caching of research papers.
- **Page-Aware Extraction:** Fast, layout-aware text extraction preserving page boundaries and formatting using PyMuPDF.
- **Structured AI Analysis:** Deep technical breakdown (TL;DR, Motivation, Methodology, Results, Limitations) using local open-weight models (Qwen 8B via Ollama) with strict Pydantic JSON schemas.

## Tech Stack
- **Language:** Python 3.10+
- **APIs:** ArXiv API (via `arxiv`)
- **PDF Processing:** `pymupdf` (Fitz)
- **Local LLM Engine:** Ollama (`qwen2.5:8b`)
- **Validation:** Pydantic v2

## Setup & Usage
1. Clone the repository:
   
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Ollama with your chosen model:
   ```bash
   ollama run qwen2.5:8b
   ```
5. Run the CLI:
   ```bash
   python main.py
   ```