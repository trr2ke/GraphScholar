"""Hybrid retriever: vector search + k-hop graph expansion.

Given a natural-language query, the retriever:
  1. Finds the most semantically similar paper abstracts via ChromaDB.
  2. Finds the most relevant community summaries via ChromaDB.
  3. Expands k hops out from the seed Paper nodes in the NetworkX graph
     to collect connected domain entities (methods, datasets, etc.).

The combined context is returned as a RetrievalResult for the generator.
"""
from dataclasses import dataclass, field

import networkx as nx

from .vectorstore import VectorStore


@dataclass
class RetrievalResult:
    """Context bundle returned to the answer generator."""
    papers: list[dict]             # from VectorStore.search_papers()
    graph_nodes: list[dict]        # expanded neighbourhood nodes
    community_summaries: list[dict]  # from VectorStore.search_communities()


class HybridRetriever:
    """Combines semantic vector search with graph neighbourhood expansion."""

    def __init__(
        self,
        graph: nx.DiGraph,
        vectorstore: VectorStore,
        n_papers: int = 5,
        n_communities: int = 3,
        k_hop: int = 2,
    ):
        """
        Args:
            graph:        The live knowledge graph from GraphBuilder.
            vectorstore:  The ChromaDB VectorStore.
            n_papers:     How many papers to retrieve via vector search.
            n_communities: How many community summaries to retrieve.
            k_hop:        Neighbourhood expansion depth from seed Paper nodes.
        """
        self.graph = graph
        self.vectorstore = vectorstore
        self.n_papers = n_papers
        self.n_communities = n_communities
        self.k_hop = k_hop

    def retrieve(self, query: str) -> RetrievalResult:
        """Run hybrid retrieval for a natural-language query.

        Args:
            query: The user's question or topic string.

        Returns:
            RetrievalResult with papers, expanded graph nodes, and community summaries.
        """
        # ── 1. Semantic search ───────────────────────────────────────
        papers = self.vectorstore.search_papers(query, self.n_papers)
        communities = self.vectorstore.search_communities(query, self.n_communities)

        # ── 2. Graph expansion from seed Paper nodes ─────────────────
        seed_node_ids = {
            f"Paper::{p['id']}"
            for p in papers
            if f"Paper::{p['id']}" in self.graph
        }
        expanded = self._expand(seed_node_ids, self.k_hop)

        return RetrievalResult(
            papers=papers,
            graph_nodes=expanded,
            community_summaries=communities,
        )

    def _expand(self, seeds: set[str], k: int) -> list[dict]:
        """Collect all nodes within k hops of the seed set.

        Args:
            seeds: Starting node IDs (Paper nodes for retrieved papers).
            k:     Number of hops to expand.

        Returns:
            List of node attribute dicts (excluding Paper and Author nodes
            to focus the context on domain-specific entities).
        """
        visited: set[str] = set(seeds)
        frontier: set[str] = set(seeds)

        for _ in range(k):
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in self.graph:
                    continue
                # Expand both successors and predecessors (undirected traversal)
                neighbours = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                next_frontier |= neighbours - visited
            visited |= next_frontier
            frontier = next_frontier

        # Return domain nodes only (skip Paper/Author — already in `papers`)
        result = []
        for nid in visited:
            if nid not in self.graph:
                continue
            data = dict(self.graph.nodes[nid])
            if data.get("type") in ("Paper", "Author"):
                continue
            data["node_id"] = nid
            result.append(data)

        return result
