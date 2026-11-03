import os
import re
from pathlib import Path
from typing import Optional
import httpx


class PDFDownloader:
    """
    Downloads and caches academic PDFs to disk to avoid repeated network requests.
    """

    def __init__(self, download_dir: str = "data/papers"):
        self.download_dir = Path(download_dir)
        # Ensure the directory exists (creates parent directories if needed)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, paper_id: str) -> str:
        """
        Replaces characters like '/', ':', or '\' that are invalid in filenames.
        Example: 'math/0101001' -> 'math_0101001.pdf'
        """
        safe_id = re.sub(r'[/\\:*?"<>|]', '_', paper_id)
        return f"{safe_id}.pdf"

    def get_local_path(self, paper_id: str) -> Path:
        """Returns the local Path where this paper would be stored."""
        return self.download_dir / self._sanitize_filename(paper_id)

    def is_cached(self, paper_id: str) -> bool:
        """Checks if the PDF is already downloaded and non-empty."""
        local_path = self.get_local_path(paper_id)
        return local_path.exists() and local_path.stat().st_size > 0

    def download_paper(self, paper_id: str, pdf_url: str) -> Optional[Path]:
        """
        Downloads the PDF from pdf_url if not already cached.
        Returns the Path to the local PDF, or None if download fails.
        """
        target_path = self.get_local_path(paper_id)

        # 1. Cache hit: Return immediately without hitting the network
        if self.is_cached(paper_id):
            print(f"[Cache Hit] Paper already downloaded at: {target_path}")
            return target_path

        # 2. Cache miss: Stream download from ArXiv
        print(f"[Downloading] Fetching PDF from {pdf_url}...")
        headers = {
            # ArXiv asks clients to include an informative User-Agent
            "User-Agent": "ResearchPaperIntelligence/1.0 (Educational AI Project)"
        }

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
                with client.stream("GET", pdf_url) as response:
                    if response.status_code != 200:
                        print(f"[Download Error] Server returned status code {response.status_code}")
                        return None

                    # Write in chunks of 8KB to avoid high memory consumption
                    with open(target_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)

            print(f"[Success] PDF saved to {target_path}")
            return target_path

        except Exception as e:
            print(f"[Download Error] Failed to download PDF: {e}")
            # If a partial/broken file was created, clean it up
            if target_path.exists():
                target_path.unlink()
            return None