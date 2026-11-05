from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import ollama

from src.models import PaperChunk


class VectorStoreManager:
    """
    Manages vector embeddings and persistent semantic search using ChromaDB and Ollama.
    """

    def __init__(
        self, 
        persist_dir: str = "data/chromadb", 
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "research_papers"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model

        # Initialize persistent Chroma client (data saves to disk)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection configured with cosine distance
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str) -> List[float]:
        """
        Calls Ollama's local embedding model to convert a string into a vector.
        """
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def is_paper_indexed(self, paper_id: str) -> bool:
        """
        Checks if this paper's chunks are already indexed in ChromaDB.
        """
        existing = self.collection.get(
            where={"paper_id": paper_id},
            limit=1
        )
        return len(existing["ids"]) > 0

    def index_chunks(self, chunks: List[PaperChunk]) -> bool:
        """
        Generates embeddings for chunks and upserts them into ChromaDB.
        """
        if not chunks:
            return False

        paper_id = chunks[0].paper_id

        # Skip if already indexed to save time and compute
        if self.is_paper_indexed(paper_id):
            print(f"[Vector Store] Paper {paper_id} is already indexed in ChromaDB.")
            return True

        print(f"[Vector Store] Generating embeddings for {len(chunks)} chunks with '{self.embedding_model}'...")

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for idx, chunk in enumerate(chunks, start=1):
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "paper_id": chunk.paper_id,
                "page_number": chunk.page_number
            })
            
            # Generate vector for each chunk
            vector = self._get_embedding(chunk.text)
            embeddings.append(vector)

            # Print a progress indicator every 15 chunks
            if idx % 15 == 0 or idx == len(chunks):
                print(f"  -> Embedded {idx}/{len(chunks)} chunks...")

        # Store in ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"[Vector Store] Successfully indexed {len(chunks)} chunks for paper {paper_id}.")
        return True

    def search_relevant_chunks(
        self, 
        query: str, 
        paper_id: str, 
        top_k: int = 4
    ) -> List[PaperChunk]:
        """
        Retrieves the top_k most relevant chunks for a query strictly within paper_id.
        """
        query_vector = self._get_embedding(query)

        # Semantic query filtered strictly by this paper's ID
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"paper_id": paper_id}
        )

        retrieved_chunks: List[PaperChunk] = []

        if not results["ids"] or not results["ids"][0]:
            return []

        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]

            retrieved_chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    paper_id=metadata["paper_id"],
                    page_number=metadata["page_number"],
                    text=text
                )
            )

        return retrieved_chunks