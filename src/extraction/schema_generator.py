"""Topic → GraphSchema via LLM (Stage 2 of the Option C schema flow).

The LLM proposes a domain-specific schema given only the research topic.
All subsequent papers are extracted into this schema, which may later
grow via SchemaEvolver (Stage 3) as new entity types are discovered.
"""
import json
import re

from ..api.base import BaseLLMProvider
from .schema import EdgeType, GraphSchema, NodeType

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SCHEMA_PROMPT = """\
You are a knowledge graph architect for a scientific literature analysis system.

Research topic: "{topic}"

Design a domain-specific knowledge graph schema to organize papers on this topic.
The schema will be used to extract structured knowledge from arXiv abstracts.

Requirements
------------
1. Include "Paper" and "Author" as the FIRST TWO node types — these are structural
   anchors present in every schema.
2. Add 3–6 DOMAIN-SPECIFIC node types tailored to this topic.
   Good examples:
     • "reinforcement learning algorithmic trading" → AI_Technique, Trading_Strategy,
       Market_Environment, Risk_Factor, Dataset
     • "CRISPR gene editing" → Gene, Protein, Disease, Organism, Treatment
     • "transformer models NLP" → Model_Architecture, Task, Dataset, Benchmark
3. Add AUTHORED_BY and CITES as the first two edge types.
4. Add 4–8 DOMAIN-SPECIFIC edge types capturing key relationships.

Return ONLY a valid JSON object — no markdown fences, no explanation, nothing else.
Use this exact structure:

{{
  "node_types": [
    {{"name": "Paper", "description": "A scientific paper", "properties": ["title", "arxiv_id", "year"]}},
    {{"name": "Author", "description": "A researcher who authored papers", "properties": ["name"]}},
    {{"name": "...", "description": "...", "properties": ["..."]}}
  ],
  "edge_types": [
    {{"name": "AUTHORED_BY", "description": "A paper was written by an author", "from_type": "Paper", "to_type": "Author"}},
    {{"name": "CITES", "description": "A paper cites another paper", "from_type": "Paper", "to_type": "Paper"}},
    {{"name": "...", "description": "...", "from_type": "...", "to_type": "..."}}
  ]
}}

Naming conventions:
  Node names  → PascalCase, underscores allowed (e.g. AI_Technique, Asset_Class)
  Edge names  → SCREAMING_SNAKE_CASE (e.g. APPLIED_TO, TRAINED_ON)
  Descriptions → one clear sentence each
"""


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class SchemaGenerator:
    """Generates an initial GraphSchema from a research topic string."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    def generate(self, topic: str) -> GraphSchema:
        """Prompt the LLM and parse its JSON response into a GraphSchema.

        Args:
            topic: Free-text research topic, e.g. "transformer models in NLP".

        Returns:
            A GraphSchema with Paper/Author anchors plus domain-specific types.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid schema JSON.
        """
        prompt = _SCHEMA_PROMPT.format(topic=topic)
        raw = self.provider.complete([{"role": "user", "content": prompt}])

        try:
            data = json.loads(_extract_json(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Schema generator returned invalid JSON.\n"
                f"Raw response:\n{raw}\nError: {exc}"
            ) from exc

        return GraphSchema(
            topic=topic,
            node_types=[NodeType(**n) for n in data["node_types"]],
            edge_types=[EdgeType(**e) for e in data["edge_types"]],
            version=1,
        )


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Strip markdown code fences if present and return the raw JSON string.

    LLMs sometimes wrap JSON in ```json ... ``` even when instructed not to.
    This function handles the common cases so callers can always call json.loads().
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        return fenced.group(1).strip()

    # Fall back: find the outermost { ... } block
    braced = re.search(r"(\{[\s\S]*\})", text)
    if braced:
        return braced.group(1).strip()

    return text.strip()
