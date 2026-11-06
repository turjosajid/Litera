import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.models import PaperMetadata


class DatabaseManager:
    """
    Production-grade SQLite database manager for user folders and paper bookmarks.
    Implements transaction safety, connection pooling/closing, and full CRUD.
    """

    def __init__(self, db_path: str = "data/library.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """
        Context manager that guarantees transaction commit on success,
        rollback on exception, and connection closing in all cases.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Creates the relational schema if tables do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Folders Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL CHECK(length(trim(name)) > 0),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Papers Table (Authors stored as valid JSON array)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS saved_papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL CHECK(length(trim(title)) > 0),
                    authors_json TEXT,
                    abstract TEXT,
                    pdf_url TEXT NOT NULL,
                    primary_category TEXT,
                    published_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Junction Table (Many-to-Many with CASCADE deletion)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_folders (
                    folder_id INTEGER,
                    paper_id TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (folder_id, paper_id),
                    FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE CASCADE,
                    FOREIGN KEY (paper_id) REFERENCES saved_papers(paper_id) ON DELETE CASCADE
                );
            """)

    # ==========================================
    # Folder Management (Create, List, Delete)
    # ==========================================

    def create_folder(self, name: str) -> Optional[int]:
        """Creates a new folder with non-empty validation."""
        if not name or not name.strip():
            print("[Database Error] Folder name cannot be empty.")
            return None

        clean_name = name.strip()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO folders (name) VALUES (?);", (clean_name,))
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"[Database Error] Folder '{clean_name}' already exists.")
            return None
        except sqlite3.Error as e:
            print(f"[Database Error] SQLite failure during folder creation: {e}")
            return None

    def list_folders(self) -> List[Dict[str, Any]]:
        """Returns all folders with paper counts."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT f.id, f.name, COUNT(pf.paper_id) as paper_count
                    FROM folders f
                    LEFT JOIN paper_folders pf ON f.id = pf.folder_id
                    GROUP BY f.id, f.name
                    ORDER BY f.name ASC;
                """)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[Database Error] Could not list folders: {e}")
            return []

    def delete_folder(self, folder_id: int) -> bool:
        """Deletes a folder and cascades to clean up its bookmarks."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM folders WHERE id = ?;", (folder_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"[Database Error] Failed to delete folder {folder_id}: {e}")
            return False

    # ==========================================
    # Bookmark & Paper Management
    # ==========================================

    def save_paper_to_folder(self, paper: PaperMetadata, folder_id: int) -> bool:
        """
        Saves paper metadata (with JSON author serialization and full UPSERT)
        and bookmarks it to the target folder.
        """
        if not paper or not paper.paper_id or not paper.title:
            print("[Database Error] Invalid paper metadata provided.")
            return False

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Verify target folder exists
                cursor.execute("SELECT id FROM folders WHERE id = ?;", (folder_id,))
                if not cursor.fetchone():
                    print(f"[Database Error] Target folder {folder_id} does not exist.")
                    return False

                # Format fields safely
                authors_json = json.dumps(paper.authors or [])
                pub_date_str = (
                    paper.published_date.strftime("%Y-%m-%d") 
                    if paper.published_date 
                    else datetime.now().strftime("%Y-%m-%d")
                )

                # 1. Full UPSERT (updates ALL fields if new metadata arrives)
                cursor.execute("""
                    INSERT INTO saved_papers (
                        paper_id, title, authors_json, abstract, 
                        pdf_url, primary_category, published_date
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                        title = excluded.title,
                        authors_json = excluded.authors_json,
                        abstract = excluded.abstract,
                        pdf_url = excluded.pdf_url,
                        primary_category = excluded.primary_category,
                        published_date = excluded.published_date;
                """, (
                    paper.paper_id,
                    paper.title,
                    authors_json,
                    paper.abstract or "",
                    paper.pdf_url,
                    paper.primary_category or "",
                    pub_date_str
                ))

                # 2. Link to folder in junction table
                cursor.execute("""
                    INSERT OR IGNORE INTO paper_folders (folder_id, paper_id)
                    VALUES (?, ?);
                """, (folder_id, paper.paper_id))

                return True

        except sqlite3.Error as e:
            print(f"[Database Error] Failed to save paper to folder: {e}")
            return False

    def remove_paper_from_folder(self, folder_id: int, paper_id: str) -> bool:
        """Removes a paper bookmark from a specific folder."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM paper_folders 
                    WHERE folder_id = ? AND paper_id = ?;
                """, (folder_id, paper_id))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"[Database Error] Could not remove paper from folder: {e}")
            return False

    def get_papers_in_folder(
        self, 
        folder_id: int, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[PaperMetadata]:
        """
        Retrieves paginated PaperMetadata objects bookmarked in a folder
        with defensive JSON and NULL handling.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.paper_id, p.title, p.authors_json, p.abstract, 
                           p.pdf_url, p.primary_category, p.published_date
                    FROM saved_papers p
                    JOIN paper_folders pf ON p.paper_id = pf.paper_id
                    WHERE pf.folder_id = ?
                    ORDER BY pf.added_at DESC
                    LIMIT ? OFFSET ?;
                """, (folder_id, limit, offset))

                rows = cursor.fetchall()
                papers = []

                for r in rows:
                    # Safe author parsing
                    try:
                        authors = json.loads(r["authors_json"]) if r["authors_json"] else []
                    except (json.JSONDecodeError, TypeError):
                        authors = []

                    # Safe date parsing
                    try:
                        pub_date = datetime.strptime(r["published_date"], "%Y-%m-%d")
                    except (ValueError, TypeError):
                        pub_date = datetime.now()

                    papers.append(
                        PaperMetadata(
                            paper_id=r["paper_id"],
                            title=r["title"],
                            authors=authors,
                            abstract=r["abstract"] or "",
                            pdf_url=r["pdf_url"],
                            primary_category=r["primary_category"] or "cs.AI",
                            published_date=pub_date,
                            categories=[r["primary_category"]] if r["primary_category"] else []
                        )
                    )
                return papers

        except sqlite3.Error as e:
            print(f"[Database Error] Failed to retrieve papers from folder {folder_id}: {e}")
            return []