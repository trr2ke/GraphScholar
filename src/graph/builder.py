"""Incremental NetworkX knowledge graph construction for GraphScholar.

The GraphBuilder maintains a directed graph (nx.DiGraph) and grows it one
paper at a time.  It owns Paper/Author node creation and delegates domain
node deduplication to NodeDeduplicator before adding extracted entities.

Node ID formula:  "{NodeType}::{canonical_label}"
  e.g.  "Paper::2310.06825"
        "Author::Jacob Devlin"
        "Model_Architecture::BERT"
"""
import json
from pathlib import Path

import networkx as nx

from ..extraction.extractor import ExtractionResult
from ..ingestion.paper import Paper
from .deduplicator import NodeDeduplicator

# Sentinel used by the Extractor when an edge endpoint is the current paper
_PAPER_SENTINEL = "__CURRENT_PAPER__"


class GraphBuilder:
    """Builds and maintains the knowledge graph incrementally."""

    def __init__(self, deduplicator: NodeDeduplicator):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.deduplicator = deduplicator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_paper(self, paper: Paper) -> str:
        """Add a Paper node plus its Author nodes and AUTHORED_BY edges.

        Args:
            paper: The paper to add.

        Returns:
            The node ID for the Paper node.
        """
        paper_nid = _nid("Paper", paper.id)

        self.graph.add_node(
            paper_nid,
            type="Paper",
            label=paper.title,
            arxiv_id=paper.id,
            url=paper.url,
            published=paper.published,
            abstract=paper.abstract[:500],   # truncated for graph storage
        )

        for author_name in paper.authors:
            author_nid = _nid("Author", author_name)
            if author_nid not in self.graph:
                self.graph.add_node(author_nid, type="Author", label=author_name)
            self.graph.add_edge(paper_nid, author_nid, type="AUTHORED_BY")

        return paper_nid

    def add_extraction(self, paper: Paper, result: ExtractionResult) -> None:
        """Add domain nodes and edges from one extraction result.

        Resolves label duplicates via the deduplicator before inserting.

        Args:
            paper:  The source paper (needed to substitute the sentinel).
            result: Extraction output from Extractor.extract().
        """
        paper_nid = _nid("Paper", paper.id)

        # ── 1. Deduplicate domain nodes ──────────────────────────────
        candidates = [(n.type, n.label) for n in result.nodes]
        label_map = self.deduplicator.resolve_batch(candidates)
        # label_map: {(type, original_label) -> canonical_label}

        # ── 2. Add domain nodes ──────────────────────────────────────
        for node in result.nodes:
            canonical = label_map.get((node.type, node.label), node.label)
            nid = _nid(node.type, canonical)
            if nid not in self.graph:
                self.graph.add_node(
                    nid,
                    type=node.type,
                    label=canonical,
                    **{k: v for k, v in node.properties.items() if isinstance(v, (str, int, float, bool))},
                )
            # Track which papers mention each node
            existing_ids = self.graph.nodes[nid].get("paper_ids", "")
            ids_set = set(existing_ids.split(",")) if existing_ids else set()
            ids_set.add(paper.id)
            self.graph.nodes[nid]["paper_ids"] = ",".join(sorted(ids_set))

        # ── 3. Add domain edges ──────────────────────────────────────
        for edge in result.edges:
            # Resolve sentinel → actual paper node ID
            from_label = edge.from_label
            to_label = edge.to_label

            if from_label == _PAPER_SENTINEL:
                from_nid = paper_nid
            else:
                canonical = label_map.get((edge.from_type, from_label), from_label)
                from_nid = _nid(edge.from_type, canonical)

            if to_label == _PAPER_SENTINEL:
                to_nid = paper_nid
            else:
                canonical = label_map.get((edge.to_type, to_label), to_label)
                to_nid = _nid(edge.to_type, canonical)

            # Only add the edge if both endpoints exist
            if from_nid in self.graph and to_nid in self.graph:
                if self.graph.has_edge(from_nid, to_nid):
                    self.graph[from_nid][to_nid]["weight"] = (
                        self.graph[from_nid][to_nid].get("weight", 1) + 1
                    )
                else:
                    self.graph.add_edge(from_nid, to_nid, type=edge.type, weight=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the graph to GraphML (readable by Gephi and networkx)."""
        nx.write_graphml(self.graph, str(path))

    @classmethod
    def load(cls, path: str | Path, deduplicator: NodeDeduplicator) -> "GraphBuilder":
        """Restore a previously saved graph."""
        builder = cls(deduplicator)
        builder.graph = nx.read_graphml(str(path))
        return builder

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def stats(self) -> str:
        return f"{self.node_count} nodes, {self.edge_count} edges"


def _nid(node_type: str, label: str) -> str:
    """Canonical node ID: '{type}::{label}'."""
    return f"{node_type}::{label}"
