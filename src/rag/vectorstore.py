"""ChromaDB vector store for GraphScholar.

Two collections:
  - "papers"              : paper abstracts + metadata, for semantic search
  - "community_summaries" : LLM-written cluster descriptions, for theme search

The embedding function delegates to whatever BaseLLMProvider is active, so
the same model handles both graph extraction and semantic indexing.
"""
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..api.base import BaseLLMProvider
from ..ingestion.paper import Paper


class _ProviderEmbedder:
    """Adapts BaseLLMProvider.embed() to ChromaDB's embedding-function interface."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        # `provider` is injected after construction (see VectorStore.__init__)
        return self._provider.embed(input)


class VectorStore:
    """Persistent ChromaDB-backed store for papers and community summaries."""

    def __init__(self, provider: BaseLLMProvider, persist_dir: str | Path = "data/vectors"):
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Build a bound embedding function for this provider
        embedder = _ProviderEmbedder()
        embedder._provider = provider

        self._papers = self._client.get_or_create_collection(
            name="papers",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )
        self._communities = self._client.get_or_create_collection(
            name="community_summaries",
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Paper operations
    # ------------------------------------------------------------------

    def add_paper(self, paper: Paper) -> None:
        """Embed the abstract and store paper metadata.

        Skips silently if this arxiv ID is already in the store.
        """
        existing = self._papers.get(ids=[paper.id])
        if existing["ids"]:
            return

        self._papers.add(
            ids=[paper.id],
            documents=[paper.abstract],
            metadatas=[{
                "title":     paper.title,
                "authors":   paper.short_authors(),
                "url":       paper.url,
                "published": paper.published,
            }],
        )

    def search_papers(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search over paper abstracts.

        Returns:
            List of dicts with keys: id, title, authors, url, published, abstract, distance.
        """
        if self._papers.count() == 0:
            return []

        n = min(n_results, self._papers.count())
        res = self._papers.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i, paper_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            hits.append({
                "id":        paper_id,
                "title":     meta.get("title", ""),
                "authors":   meta.get("authors", ""),
                "url":       meta.get("url", ""),
                "published": meta.get("published", ""),
                "abstract":  res["documents"][0][i],
                "distance":  res["distances"][0][i],
            })
        return hits

    # ------------------------------------------------------------------
    # Community summary operations
    # ------------------------------------------------------------------

    def add_community_summary(
        self, community_id: str, summary: str, node_labels: list[str]
    ) -> None:
        """Store or update a community summary.

        Upserts so re-running community detection replaces stale summaries.
        """
        self._communities.upsert(
            ids=[community_id],
            documents=[summary],
            metadatas=[{"node_count": len(node_labels),
                        "nodes": ", ".join(node_labels[:20])}],
        )

    def search_communities(self, query: str, n_results: int = 3) -> list[dict]:
        """Semantic search over community summaries.

        Returns:
            List of dicts with keys: id, summary, node_count, nodes, distance.
        """
        if self._communities.count() == 0:
            return []

        n = min(n_results, self._communities.count())
        res = self._communities.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i, comm_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            hits.append({
                "id":         comm_id,
                "summary":    res["documents"][0][i],
                "node_count": meta.get("node_count", 0),
                "nodes":      meta.get("nodes", ""),
                "distance":   res["distances"][0][i],
            })
        return hits
