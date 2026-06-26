"""Central prompt-template registry for GraphScholar.

All prompt strings live here so they can be reviewed, tuned, and cited
in the project write-up without hunting through multiple source files.
"""

# Used by SchemaGenerator
SCHEMA_GENERATION = """\
You are a knowledge graph architect for a scientific literature analysis system.

Research topic: "{topic}"

Design a domain-specific knowledge graph schema to organize papers on this topic.
The schema must capture the key entities and relationships most useful for
understanding this research area.

Requirements:
1. Include "Paper" and "Author" as the FIRST TWO node types (structural anchors).
2. Add 3–6 DOMAIN-SPECIFIC node types tailored to this topic.
3. Add AUTHORED_BY and CITES as the first two edge types.
4. Add 4–8 DOMAIN-SPECIFIC edge types capturing key relationships.

Return ONLY a valid JSON object — no markdown fences, no explanation:

{{
  "node_types": [
    {{"name": "Paper", "description": "A scientific paper", "properties": ["title", "arxiv_id", "year"]}},
    {{"name": "Author", "description": "A researcher who authored papers", "properties": ["name"]}},
    {{"name": "...", "description": "...", "properties": ["..."]}}
  ],
  "edge_types": [
    {{"name": "AUTHORED_BY", "description": "Paper written by Author", "from_type": "Paper", "to_type": "Author"}},
    {{"name": "CITES", "description": "Paper cites another Paper", "from_type": "Paper", "to_type": "Paper"}},
    {{"name": "...", "description": "...", "from_type": "...", "to_type": "..."}}
  ]
}}

Node names: PascalCase.  Edge names: SCREAMING_SNAKE_CASE.
Keep each description to one clear sentence.
"""

# Used by Extractor
ENTITY_EXTRACTION = """\
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

# Used by SchemaEvolver
SCHEMA_EVOLUTION = """\
You are evolving a knowledge graph schema based on entities found in scientific papers
that didn't fit the current schema.

Current schema topic: {topic}
Current node types: {current_nodes}
Current edge types: {current_edges}

Entities that didn't fit the current schema:
{unfit_list}

Propose additions to accommodate these entities. Return ONLY valid JSON:

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
- Only propose genuinely new types; do not duplicate existing ones.
- New edge from_type/to_type must exist in the current schema or proposed_node_types.
- If no additions are warranted, return {{"proposed_node_types": [], "proposed_edge_types": []}}.
"""

# Used by CommunityDetector
COMMUNITY_SUMMARY = """\
The following entities form a cluster in a scientific knowledge graph about "{topic}".

Entities in this cluster:
{entity_list}

Write a 2-3 sentence summary describing what this cluster represents: the common theme
connecting these entities and why they belong together.  Be specific and concise.
"""

# Used by AnswerGenerator
ANSWER_SYSTEM = """\
You are a research assistant with access to a structured knowledge graph built from
scientific papers.  Answer questions accurately using only the provided context.
Cite every factual claim with the arXiv ID of the supporting paper in square
brackets, e.g. [2310.06825].  If the context does not contain enough information
to answer confidently, say so rather than speculating.
"""
