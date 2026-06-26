"""Citation-grounded answer generator for GraphScholar.

Formats retrieval context (papers, graph nodes, community summaries) into
a structured prompt and calls the LLM to produce an answer with arXiv citations.
"""
from ..api.base import BaseLLMProvider
from .retriever import RetrievalResult

_SYSTEM_PROMPT = """\
You are a research assistant with access to a structured knowledge graph built from
scientific papers.  Answer questions accurately using only the provided context.
Cite every factual claim with the arXiv ID of the supporting paper in square
brackets, e.g. [2310.06825].  If the context does not contain enough information
to answer confidently, say so rather than speculating.
"""

_CONTEXT_TEMPLATE = """\
=== RELEVANT PAPERS ===
{papers_block}

=== RELATED KNOWLEDGE GRAPH ENTITIES ===
{nodes_block}

=== THEMATIC CLUSTERS ===
{communities_block}

=== QUESTION ===
{query}
"""


class AnswerGenerator:
    """Generates cited answers from retrieval context and conversation history."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    def generate(
        self,
        query: str,
        retrieval: RetrievalResult,
        history: list[dict],
    ) -> str:
        """Produce a cited answer.

        Args:
            query:     The user's question.
            retrieval: Context from HybridRetriever.retrieve().
            history:   Prior conversation turns as {"role", "content"} dicts.

        Returns:
            Answer string with arXiv citation markers.
        """
        context = _CONTEXT_TEMPLATE.format(
            papers_block=_format_papers(retrieval.papers),
            nodes_block=_format_nodes(retrieval.graph_nodes),
            communities_block=_format_communities(retrieval.community_summaries),
            query=query,
        )
        messages = [
            {"role": "user" if h["role"] == "user" else "assistant",
             "content": h["content"]}
            for h in history
        ] + [{"role": "user", "content": context}]

        return self.provider.complete(messages, system=_SYSTEM_PROMPT, max_tokens=1024)


# ---------------------------------------------------------------------------
# Context formatters
# ---------------------------------------------------------------------------

def _format_papers(papers: list[dict]) -> str:
    if not papers:
        return "No papers retrieved."
    lines = []
    for p in papers:
        lines.append(
            f"[{p['id']}] {p['title']}\n"
            f"  Authors: {p['authors']}  ({p['published']})\n"
            f"  Abstract: {p['abstract'][:300]}..."
        )
    return "\n\n".join(lines)


def _format_nodes(nodes: list[dict]) -> str:
    if not nodes:
        return "No graph entities retrieved."
    lines = []
    for n in nodes:
        ntype = n.get("type", "?")
        label = n.get("label", n.get("node_id", "?"))
        papers = n.get("paper_ids", "")
        cite = f"  [mentioned in: {papers}]" if papers else ""
        lines.append(f"• {ntype}: {label}{cite}")
    return "\n".join(lines)


def _format_communities(communities: list[dict]) -> str:
    if not communities:
        return "No thematic clusters retrieved."
    lines = []
    for c in communities:
        lines.append(
            f"Cluster {c['id']} ({c.get('node_count', '?')} entities):\n"
            f"  {c['summary']}"
        )
    return "\n\n".join(lines)
