"""Schema evolution for GraphScholar (Option C — topic-seeded + incrementally evolving).

The SchemaEvolver accumulates entities that don't fit the current schema
during extraction, then proposes new NodeType/EdgeType additions to the user.
The user approves or rejects proposals in the GUI before the schema is updated.
"""
import json
import re

from ..api.base import BaseLLMProvider
from .schema import EdgeType, GraphSchema, NodeType


_EVOLUTION_PROMPT = """\
You are evolving a knowledge graph schema based on entities found in scientific papers
that didn't fit the current schema.

Current schema topic: {topic}

Current node types: {current_nodes}
Current edge types: {current_edges}

Entities that didn't fit the current schema:
{unfit_list}

Propose additions to the schema to accommodate these entities. Return ONLY valid JSON:

{{
  "proposed_node_types": [
    {{"name": "<PascalCase>", "description": "<one sentence>", "properties": ["<attr>"]}}
  ],
  "proposed_edge_types": [
    {{"name": "<SCREAMING_SNAKE_CASE>", "description": "<one sentence>",
      "from_type": "<NodeType>", "to_type": "<NodeType>"}}
  ]
}}

Rules:
- Only propose types that are genuinely new — do not duplicate existing ones.
- Consolidate similar unfit entities into one new type where appropriate.
- New node names must be PascalCase; new edge names SCREAMING_SNAKE_CASE.
- from_type and to_type in proposed edges must reference types that ALREADY EXIST
  in the current schema or appear in proposed_node_types above.
- If no additions are warranted, return {{"proposed_node_types": [], "proposed_edge_types": []}}.
"""


class SchemaEvolver:
    """Collects unfit entities across papers and proposes schema additions."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self._pending: list[dict] = []

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def accumulate(self, unfit_entities: list[dict]) -> None:
        """Add unfit entities from one paper's extraction to the pending queue."""
        self._pending.extend(unfit_entities)

    @property
    def pending_count(self) -> int:
        """Number of unfit entities waiting for a proposal pass."""
        return len(self._pending)

    def clear(self) -> None:
        """Clear the pending queue after proposals have been presented to the user."""
        self._pending.clear()

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def propose_additions(
        self, schema: GraphSchema
    ) -> tuple[list[NodeType], list[EdgeType]]:
        """Ask the LLM to propose new types based on accumulated unfit entities.

        Args:
            schema: The current schema (used to avoid duplicate proposals).

        Returns:
            Tuple of (proposed_node_types, proposed_edge_types).
            Returns empty lists if there is nothing to propose.
        """
        if not self._pending:
            return [], []

        unfit_lines = "\n".join(
            f"  • \"{e.get('text', '')}\" — suggested type: {e.get('suggested_type', '?')} "
            f"({e.get('description', '')})"
            for e in self._pending
        )

        prompt = _EVOLUTION_PROMPT.format(
            topic=schema.topic,
            current_nodes=", ".join(schema.node_names()),
            current_edges=", ".join(schema.edge_names()),
            unfit_list=unfit_lines,
        )

        raw = self.provider.complete([{"role": "user", "content": prompt}])

        try:
            data = json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            return [], []

        new_nodes = [
            NodeType(**n)
            for n in data.get("proposed_node_types", [])
            if n.get("name") and n["name"] not in schema.node_names()
        ]
        new_edges = [
            EdgeType(**e)
            for e in data.get("proposed_edge_types", [])
            if e.get("name") and e["name"] not in schema.edge_names()
        ]
        return new_nodes, new_edges


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        return fenced.group(1).strip()
    braced = re.search(r"(\{[\s\S]*\})", text)
    if braced:
        return braced.group(1).strip()
    return text.strip()
