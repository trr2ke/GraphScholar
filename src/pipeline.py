"""Top-level pipeline orchestrator for GraphScholar.

Splits the workflow into two phases:
  Phase 1 — generate_schema(topic)  → GraphSchema  (call once; user reviews in GUI)
  Phase 2 — ingest(topic, n_papers) → Generator[PipelineEvent]  (streams status to GUI)

After ingestion, answer(query) runs the full RAG chain and returns a cited answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import networkx as nx

from .api.base import BaseLLMProvider
from .chat.session import ChatSession
from .extraction.extractor import Extractor
from .extraction.schema import EdgeType, GraphSchema, NodeType
from .extraction.schema_evolver import SchemaEvolver
from .extraction.schema_generator import SchemaGenerator
from .graph.builder import GraphBuilder
from .graph.community import CommunityDetector
from .graph.deduplicator import NodeDeduplicator
from .graph.visualizer import GraphVisualizer
from .ingestion.arxiv_client import ArxivClient
from .rag.generator import AnswerGenerator
from .rag.retriever import HybridRetriever, RetrievalResult
from .rag.vectorstore import VectorStore


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineEvent:
    """Status event yielded by Pipeline.ingest().

    type values
    -----------
    "paper_processed"    – one paper extracted and added to the graph
    "schema_proposal"    – evolver found candidate new types; GUI shows accept/reject
    "community_detected" – Louvain detection + LLM summarisation complete
    "done"               – all papers processed; pipeline finished
    "error"              – non-fatal error; pipeline continues
    """
    type: str
    payload: dict = field(default_factory=dict)


# How often to run the schema evolver (every N papers)
_EVOLUTION_BATCH = 5


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Full GraphScholar pipeline for one research session.

    Lifecycle
    ---------
    1. ``pipe = Pipeline(provider)``
    2. ``schema = pipe.generate_schema(topic)``   # Phase 1 — LLM proposes schema
    3. *(user reviews/edits schema in GUI)*
    4. ``pipe.set_schema(schema)``                # apply any edits
    5. ``for event in pipe.ingest(topic, n):``    # Phase 2 — stream events
    6. ``answer = pipe.answer(query)``            # RAG chat
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        persist_dir: str | Path = "data/vectors",
        similarity_threshold: float = 0.85,
    ):
        """
        Args:
            provider:             Active LLM provider (Claude, OpenAI, etc.).
            persist_dir:          ChromaDB persistence directory.
            similarity_threshold: Cosine-sim cutoff for node deduplication.
        """
        self._provider = provider

        self._arxiv       = ArxivClient()
        self._schema_gen  = SchemaGenerator(provider)
        self._extractor   = Extractor(provider)
        self._evolver     = SchemaEvolver(provider)
        self._dedup       = NodeDeduplicator(provider, threshold=similarity_threshold)
        self._builder     = GraphBuilder(self._dedup)
        self._visualizer  = GraphVisualizer()
        self._vectorstore = VectorStore(provider, persist_dir)
        self._community   = CommunityDetector(provider, self._vectorstore)
        self._session     = ChatSession()

        self._schema: GraphSchema | None = None
        self._last_retrieval: RetrievalResult | None = None

    # ------------------------------------------------------------------
    # Phase 1 — Schema generation
    # ------------------------------------------------------------------

    def generate_schema(self, topic: str) -> GraphSchema:
        """Ask the LLM to propose an initial GraphSchema for this topic.

        Args:
            topic: Research topic string entered by the user.

        Returns:
            Proposed GraphSchema (always includes Paper and Author node types).
        """
        self._schema = self._schema_gen.generate(topic)
        return self._schema

    def set_schema(self, schema: GraphSchema) -> None:
        """Replace the active schema (called after the user edits the Schema tab)."""
        self._schema = schema

    # ------------------------------------------------------------------
    # Phase 2 — Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self, topic: str, n_papers: int = 10
    ) -> Generator[PipelineEvent, None, None]:
        """Fetch papers, extract the knowledge graph, and run community detection.

        Yields a PipelineEvent after each paper is processed, plus events for
        schema evolution proposals, community detection, and final completion.

        Args:
            topic:    Research topic (arXiv query string and community context).
            n_papers: Maximum papers to fetch and process.
        """
        if self._schema is None:
            self._schema = self.generate_schema(topic)

        try:
            papers = self._arxiv.search(topic, max_results=n_papers)
        except Exception as exc:
            yield PipelineEvent("error", {"message": f"arXiv fetch failed: {exc}"})
            return

        total = len(papers)

        for i, paper in enumerate(papers):
            # Index abstract in vector store (non-fatal if it fails)
            try:
                self._vectorstore.add_paper(paper)
            except Exception:
                pass

            # Add Paper + Author nodes to the graph
            self._builder.add_paper(paper)

            # LLM extraction: entities + relationships
            try:
                result = self._extractor.extract(paper, self._schema)
            except Exception as exc:
                yield PipelineEvent(
                    "error",
                    {"message": f"Extraction failed for {paper.id}: {exc}"},
                )
                continue

            # Queue unfit entities for schema evolution
            self._evolver.accumulate(result.unfit_entities)

            # Merge extracted nodes/edges into the graph (with dedup)
            self._builder.add_extraction(paper, result)

            html = self._visualizer.render(self._builder.graph)

            yield PipelineEvent("paper_processed", {
                "paper_index":  i + 1,
                "total_papers": total,
                "paper_title":  paper.title,
                "paper_id":     paper.id,
                "graph_html":   html,
                "node_count":   self._builder.node_count,
                "edge_count":   self._builder.edge_count,
            })

            # Propose schema additions every EVOLUTION_BATCH papers
            if (i + 1) % _EVOLUTION_BATCH == 0 and self._evolver.pending_count > 0:
                yield from self._emit_proposals()

        # Final evolution pass for any remaining unfit entities
        if self._evolver.pending_count > 0:
            yield from self._emit_proposals()

        # Community detection and LLM summarisation
        summaries = self._community.detect_and_summarise(self._builder.graph, topic)
        html = self._visualizer.render(self._builder.graph)
        yield PipelineEvent("community_detected", {
            "n_communities": len(summaries),
            "graph_html":    html,
        })

        yield PipelineEvent("done", {
            "node_count": self._builder.node_count,
            "edge_count": self._builder.edge_count,
        })

    def _emit_proposals(self) -> Generator[PipelineEvent, None, None]:
        """Run SchemaEvolver and yield a schema_proposal event if types were found."""
        try:
            new_nodes, new_edges = self._evolver.propose_additions(self._schema)
        except Exception:
            self._evolver.clear()
            return
        self._evolver.clear()

        if new_nodes or new_edges:
            yield PipelineEvent("schema_proposal", {
                "proposed_nodes": [
                    {"name": n.name, "description": n.description,
                     "properties": n.properties}
                    for n in new_nodes
                ],
                "proposed_edges": [
                    {"name": e.name, "description": e.description,
                     "from_type": e.from_type, "to_type": e.to_type}
                    for e in new_edges
                ],
            })

    # ------------------------------------------------------------------
    # Schema evolution — apply user decisions from the GUI
    # ------------------------------------------------------------------

    def apply_evolution(
        self,
        accepted_nodes: list[dict],
        accepted_edges: list[dict],
    ) -> None:
        """Merge accepted schema additions into the active schema.

        Args:
            accepted_nodes: Dicts with keys name, description, properties.
            accepted_edges: Dicts with keys name, description, from_type, to_type.
        """
        changed = False
        for n in accepted_nodes:
            if n["name"] not in self._schema.node_names():
                self._schema.node_types.append(
                    NodeType(
                        name=n["name"],
                        description=n.get("description", ""),
                        properties=n.get("properties", []),
                    )
                )
                changed = True
        for e in accepted_edges:
            if e["name"] not in self._schema.edge_names():
                self._schema.edge_types.append(
                    EdgeType(
                        name=e["name"],
                        description=e.get("description", ""),
                        from_type=e.get("from_type", ""),
                        to_type=e.get("to_type", ""),
                    )
                )
                changed = True
        if changed:
            self._schema.version += 1

    # ------------------------------------------------------------------
    # RAG chat
    # ------------------------------------------------------------------

    def answer(self, query: str) -> str:
        """Retrieve context and generate a citation-grounded answer.

        Args:
            query: The user's natural-language research question.

        Returns:
            Answer string with arXiv ID citation markers, e.g. [2310.06825].
        """
        retriever = HybridRetriever(self._builder.graph, self._vectorstore)
        self._last_retrieval = retriever.retrieve(query)
        generator = AnswerGenerator(self._provider)
        response = generator.generate(
            query, self._last_retrieval, self._session.get_history()
        )
        self._session.add_turn(query, response)
        return response

    @property
    def last_retrieval(self) -> RetrievalResult | None:
        """The RetrievalResult from the most recent answer() call."""
        return self._last_retrieval

    # ------------------------------------------------------------------
    # Persistence + convenience
    # ------------------------------------------------------------------

    def save_graph(self, path: str | Path) -> None:
        """Serialise the knowledge graph to GraphML (readable by Gephi)."""
        self._builder.save(path)

    def get_graph_html(self) -> str:
        """Render the current graph to a self-contained PyVis HTML string."""
        return self._visualizer.render(self._builder.graph)

    @property
    def graph(self) -> nx.DiGraph:
        """The live NetworkX knowledge graph."""
        return self._builder.graph
