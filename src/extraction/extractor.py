"""LLM-based entity and relationship extractor for GraphScholar.

Given a Paper and a GraphSchema, the extractor asks the LLM to identify
domain entities (nodes) and relationships (edges) present in the abstract,
returning them as structured dataclasses.

Special sentinel "__CURRENT_PAPER__" is used in edges that connect a domain
entity to the current paper — the GraphBuilder substitutes the real node ID.
"""
import json
import re
from dataclasses import dataclass, field

from ..api.base import BaseLLMProvider
from ..ingestion.paper import Paper
from .schema import GraphSchema


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExtractedNode:
    """A single entity extracted from a paper abstract."""
    type: str          # must match a NodeType.name in the schema
    label: str         # exact or close text from the abstract
    properties: dict = field(default_factory=dict)


@dataclass
class ExtractedEdge:
    """A directional relationship between two entities."""
    type: str          # must match an EdgeType.name in the schema
    from_type: str
    from_label: str    # use "__CURRENT_PAPER__" to reference the current paper
    to_type: str
    to_label: str      # use "__CURRENT_PAPER__" to reference the current paper


@dataclass
class ExtractionResult:
    """Full extraction output for one paper."""
    paper_id: str
    nodes: list[ExtractedNode]
    edges: list[ExtractedEdge]
    unfit_entities: list[dict]   # entities flagged for schema evolution


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are extracting entities and relationships from a scientific paper to build a knowledge graph.

SCHEMA — extract ONLY into these types
=======================================
Node types:
{node_types}

Edge types:
{edge_types}

PAPER
=====
ID: {paper_id}
Title: {title}
Authors: {authors}
Abstract:
{abstract}

INSTRUCTIONS
============
Return ONLY valid JSON with no markdown fences.
Do NOT include Paper or Author nodes — those are added automatically.
Use the sentinel "__CURRENT_PAPER__" as the label whenever an edge endpoint
is the current paper (from_type = "Paper" or to_type = "Paper").

{{
  "nodes": [
    {{"type": "<NodeType>", "label": "<name from abstract>", "properties": {{}}}}
  ],
  "edges": [
    {{"type": "<EdgeType>", "from_type": "<NodeType>", "from_label": "<label>",
      "to_type": "<NodeType>", "to_label": "<label>"}}
  ],
  "unfit_entities": [
    {{"text": "<entity>", "suggested_type": "<PascalCase>", "description": "<one sentence>"}}
  ]
}}

Rules:
- Extract 2–6 domain nodes and 2–8 edges per paper.
- Both edge endpoints must be in the nodes array, or one must be "__CURRENT_PAPER__".
- Add to unfit_entities any entity that clearly doesn't fit any existing node type.
- If nothing is relevant, return {{"nodes": [], "edges": [], "unfit_entities": []}}.
"""


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class Extractor:
    """Extracts domain entities and relationships from paper abstracts."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    def extract(self, paper: Paper, schema: GraphSchema) -> ExtractionResult:
        """Run extraction for a single paper.

        Args:
            paper: The paper whose abstract will be processed.
            schema: Current knowledge graph schema defining valid types.

        Returns:
            ExtractionResult with nodes, edges, and any flagged unfit entities.
        """
        prompt = _build_prompt(paper, schema)
        raw = self.provider.complete([{"role": "user", "content": prompt}])

        try:
            data = json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            # Return empty result rather than crashing the pipeline
            return ExtractionResult(
                paper_id=paper.id, nodes=[], edges=[], unfit_entities=[]
            )

        valid_node_types = schema.node_names()
        valid_edge_types = schema.edge_names()

        nodes = [
            ExtractedNode(
                type=n["type"],
                label=n.get("label", "").strip(),
                properties=n.get("properties", {}),
            )
            for n in data.get("nodes", [])
            if n.get("type") in valid_node_types and n.get("label")
        ]

        edges = [
            ExtractedEdge(
                type=e["type"],
                from_type=e.get("from_type", ""),
                from_label=e.get("from_label", "").strip(),
                to_type=e.get("to_type", ""),
                to_label=e.get("to_label", "").strip(),
            )
            for e in data.get("edges", [])
            if e.get("type") in valid_edge_types
        ]

        return ExtractionResult(
            paper_id=paper.id,
            nodes=nodes,
            edges=edges,
            unfit_entities=data.get("unfit_entities", []),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(paper: Paper, schema: GraphSchema) -> str:
    """Format the extraction prompt with schema and paper details."""
    node_lines = "\n".join(
        f"  • {n.name}: {n.description}"
        for n in schema.node_types
        if n.name not in ("Paper", "Author")
    )
    edge_lines = "\n".join(
        f"  • {e.name} ({e.from_type} -> {e.to_type}): {e.description}"
        for e in schema.edge_types
    )
    return _EXTRACTION_PROMPT.format(
        node_types=node_lines,
        edge_types=edge_lines,
        paper_id=paper.id,
        title=paper.title,
        authors=paper.short_authors(),
        abstract=paper.abstract,
    )


def _extract_json(text: str) -> str:
    """Strip markdown code fences and return the raw JSON string."""
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        return fenced.group(1).strip()
    braced = re.search(r"(\{[\s\S]*\})", text)
    if braced:
        return braced.group(1).strip()
    return text.strip()
