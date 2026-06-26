"""Community detection and summarisation for GraphScholar.

Runs Louvain community detection on the knowledge graph (converted to
undirected for this purpose) and asks the LLM to write a 2-3 sentence
summary for each detected cluster.  Summaries are stored in ChromaDB so
the retriever can search them semantically.
"""
import networkx as nx

from ..api.base import BaseLLMProvider
from ..rag.vectorstore import VectorStore

try:
    import community as community_louvain   # python-louvain
except ImportError:
    community_louvain = None                # handle gracefully below

_MIN_NODES_FOR_DETECTION = 4  # skip detection on very small graphs

_SUMMARY_PROMPT = """\
The following entities form a cluster in a scientific knowledge graph about "{topic}".

Entities in this cluster:
{entity_list}

Write a 2-3 sentence summary describing what this cluster represents: the common theme
connecting these entities and why they belong together.  Be specific and concise.
"""


class CommunityDetector:
    """Detects communities in the graph and stores LLM summaries in ChromaDB."""

    def __init__(self, provider: BaseLLMProvider, vectorstore: VectorStore):
        self.provider = provider
        self.vectorstore = vectorstore

    def detect_and_summarise(
        self, graph: nx.DiGraph, topic: str
    ) -> dict[int, str]:
        """Run Louvain detection and generate one summary per community.

        Args:
            graph: Current knowledge graph (will be converted to undirected).
            topic: Research topic string used to contextualise summaries.

        Returns:
            Mapping {community_id: summary_text}.
            Returns {} if the graph is too small or python-louvain is missing.
        """
        if community_louvain is None:
            return {}
        if graph.number_of_nodes() < _MIN_NODES_FOR_DETECTION:
            return {}

        undirected = graph.to_undirected()
        partition: dict[str, int] = community_louvain.best_partition(undirected)

        # Group node IDs by community
        communities: dict[int, list[str]] = {}
        for node_id, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node_id)

        # Write community IDs back onto graph nodes for the visualiser
        for node_id, comm_id in partition.items():
            if node_id in graph.nodes:
                graph.nodes[node_id]["community"] = comm_id

        summaries: dict[int, str] = {}
        for comm_id, node_ids in communities.items():
            labels = [
                f"{graph.nodes[n].get('type', '?')}: {graph.nodes[n].get('label', n)}"
                for n in node_ids
                if n in graph.nodes
            ]
            summary = self._summarise(labels, topic)
            summaries[comm_id] = summary
            self.vectorstore.add_community_summary(
                community_id=str(comm_id),
                summary=summary,
                node_labels=[graph.nodes[n].get("label", n)
                             for n in node_ids if n in graph.nodes],
            )

        return summaries

    def _summarise(self, entity_descriptions: list[str], topic: str) -> str:
        """Ask the LLM to describe one community cluster."""
        entity_list = "\n".join(f"  • {e}" for e in entity_descriptions[:30])
        prompt = _SUMMARY_PROMPT.format(topic=topic, entity_list=entity_list)
        return self.provider.complete([{"role": "user", "content": prompt}],
                                      max_tokens=256)
