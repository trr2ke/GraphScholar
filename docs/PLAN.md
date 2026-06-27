# GraphScholar — Implementation Plan

## Context

GraphScholar is a greenfield grad school project (repo: `C:\Users\e40028265\Documents\GitHub\GraphScholar`).
Currently only a README exists. The goal is an end-to-end pipeline that:
1. Takes a user research topic
2. Fetches papers from arXiv
3. Proposes a domain-specific knowledge graph schema (via LLM), then evolves it as papers are processed
4. Extracts entities/relationships into that schema and builds a live knowledge graph
5. Indexes the graph with GraphRAG (community detection + vector embeddings)
6. Provides a conversational interface with citation-grounded answers
7. Shows the graph being built in real-time in a Gradio GUI

Key decisions made:
- **Knowledge graph** (not tree) — cross-paper connections (shared methods, authors, datasets) are first-class
- **Topic-seeded schema, Option C** — LLM proposes schema at topic entry, schema evolves incrementally as new entity types emerge during extraction; user approves additions in GUI
- **Gradio** for GUI, with PyVis HTML embedded for interactive graph visualization
- **NetworkX** for in-memory graph (persisted to .graphml), **ChromaDB** for vector store
- **Configurable LLM provider** — Claude first, with abstraction layer for OpenAI, GLM4, Kimi, etc.
- Well-commented throughout; clear structure for thesis/report use

---

## Directory Structure

```
graphscholar/
├── src/
│   ├── api/                      # LLM provider abstraction
│   │   ├── base.py               # BaseLLMProvider ABC (complete + embed)
│   │   ├── claude.py             # Anthropic Claude implementation
│   │   ├── openai_provider.py    # OpenAI + compatible APIs (GLM4, Kimi share this)
│   │   └── factory.py            # get_provider(name, config) → BaseLLMProvider
│   ├── ingestion/
│   │   ├── arxiv_client.py       # arXiv search + fetch (arxiv library)
│   │   └── paper.py              # Paper dataclass (id, title, abstract, authors, url)
│   ├── extraction/
│   │   ├── schema.py             # NodeType, EdgeType, GraphSchema dataclasses
│   │   ├── schema_generator.py   # Topic → initial GraphSchema via LLM
│   │   ├── schema_evolver.py     # Flags unfit entities, proposes schema additions
│   │   └── extractor.py          # Paper + GraphSchema → ExtractedNode/ExtractedEdge
│   ├── graph/
│   │   ├── builder.py            # Incremental NetworkX graph construction
│   │   ├── deduplicator.py       # Embedding cosine-sim dedup (merges "LSTM" / "Long Short-Term Memory")
│   │   ├── community.py          # python-louvain community detection + LLM summaries
│   │   └── visualizer.py         # NetworkX → PyVis HTML (nodes colored by type, sized by degree)
│   ├── rag/
│   │   ├── vectorstore.py        # ChromaDB wrapper (collections: papers, community_summaries)
│   │   ├── retriever.py          # Hybrid retrieval: vector search + k-hop graph expansion
│   │   └── generator.py          # Retrieved context → grounded answer via LLM
│   ├── chat/
│   │   ├── session.py            # Chat history + context window management
│   │   └── prompts.py            # All prompt templates (schema gen, extraction, answer gen)
│   └── pipeline.py               # Top-level orchestrator — async generator yielding status updates
├── gui/
│   └── app.py                    # Gradio app (4 tabs; calls pipeline.py via yield)
├── data/
│   ├── papers/                   # Cached paper metadata as JSON
│   ├── graphs/                   # Saved .graphml files per session
│   └── vectors/                  # ChromaDB persistence directory
├── docs/                         # STRUCTURE.md written at end of project
├── notebooks/                    # Jupyter exploration / prototyping
├── config/
│   └── config.yaml               # Provider name, model IDs, n_papers, similarity threshold
├── .env.example                  # API key template
└── requirements.txt
```

---

## Core Data Models (`src/extraction/schema.py`)

```python
@dataclass
class NodeType:
    name: str            # e.g. "AI_Technique"
    description: str     # shown in GUI and used in extraction prompt
    properties: list[str]  # e.g. ["name", "subcategory", "year_introduced"]

@dataclass
class EdgeType:
    name: str            # e.g. "APPLIED_TO"
    description: str
    from_type: str       # must match a NodeType.name
    to_type: str

@dataclass
class GraphSchema:
    topic: str
    node_types: list[NodeType]   # always includes Paper, Author
    edge_types: list[EdgeType]
    version: int = 1             # increments on each approved evolution
```

---

## LLM Provider Abstraction (`src/api/`)

`BaseLLMProvider` ABC with two methods:
- `complete(messages: list[dict], **kwargs) -> str`
- `embed(texts: list[str]) -> list[list[float]]`

Implementations:
- `ClaudeProvider` — uses `anthropic` SDK; model configurable (default: `claude-opus-4-8`)
- `OpenAIProvider` — uses `openai` SDK; covers OpenAI, Kimi, GLM4 (all have OpenAI-compatible endpoints — just swap `base_url` in config)

`factory.py` exposes `get_provider(name: str, config: dict) -> BaseLLMProvider`.

Provider and model are set in `config/config.yaml` and overridable via GUI dropdown.

---

## Schema Flow (Option C — seeded + evolving)

### Initial generation (`schema_generator.py`)
1. User enters topic in GUI
2. LLM prompt: *"Given the research topic '{topic}', propose a knowledge graph schema for organizing academic papers. Return JSON with node_types and edge_types arrays. Always include Paper and Author node types."*
3. Parse JSON → `GraphSchema`; display in GUI Schema tab as editable table
4. User approves/edits before ingestion starts

### Evolution during extraction (`schema_evolver.py`)
1. During paper extraction, if the LLM encounters an entity that doesn't fit any current NodeType, it returns a flag: `{"unfit": true, "suggested_type": "...", "description": "..."}`
2. After each batch of N papers, the evolver aggregates flagged entities and prompts LLM: *"These entities didn't fit the current schema. Propose additions."*
3. Proposed additions are shown in GUI Pipeline tab as cards — user clicks Accept/Reject
4. Accepted types are added to `GraphSchema` (version incremented); flagged entities are re-extracted with updated schema

---

## GUI — Gradio (`gui/app.py`)

Four tabs:

| Tab | Contents |
|-----|----------|
| **Setup** | Topic input, provider dropdown (Claude/OpenAI/GLM4/Kimi), API key fields, model name, n_papers slider, Start button |
| **Schema** | Proposed schema displayed as two tables (node types, edge types); add/remove rows; Approve & Ingest button |
| **Pipeline** | Progress bar, paper count, schema evolution proposals (Accept/Reject cards), live graph (PyVis HTML via `gr.HTML`, refreshed per paper) |
| **Chat** | Conversation interface (`gr.Chatbot`), sources panel listing cited papers and graph nodes |

Graph visualization: PyVis renders to HTML string; nodes colored by NodeType, sized by degree centrality; edges labeled by EdgeType. Regenerated after each paper and pushed to `gr.HTML`.

---

## Implementation Stages

### Stage 1 — Skeleton + LLM Providers
- Create all directories and `__init__.py` files
- `requirements.txt`, `config/config.yaml`, `.env.example`, `.gitignore`
- Implement `src/api/`: base, claude, openai_provider, factory
- **Verify**: `python -c "from src.api.factory import get_provider; p = get_provider('claude', {}); print(p.complete([{'role':'user','content':'hello'}]))"`

### Stage 2 — Ingestion + Schema Generation
- `src/ingestion/arxiv_client.py` — search by topic, return list of `Paper`
- `src/extraction/schema.py` — dataclasses + JSON serialization
- `src/extraction/schema_generator.py` — topic → `GraphSchema` via LLM
- **Verify**: Enter topic "transformer models NLP" → print proposed schema to console

### Stage 3 — Extraction + Graph Construction
- `src/extraction/extractor.py` — Paper + schema → extracted nodes/edges (LLM, returns structured JSON)
- `src/extraction/schema_evolver.py` — flag handling + proposal generation
- `src/graph/builder.py` — add extracted nodes/edges to NetworkX graph; maintain node registry
- `src/graph/deduplicator.py` — cosine similarity dedup on new nodes vs existing same-type nodes
- `src/graph/visualizer.py` — NetworkX → PyVis HTML string
- **Verify**: Process 5 papers, open PyVis HTML in browser, inspect graph

### Stage 4 — RAG Layer
- `src/rag/vectorstore.py` — ChromaDB collections for paper abstracts and community summaries
- `src/graph/community.py` — python-louvain community detection; LLM summary per community; store in ChromaDB
- `src/rag/retriever.py` — vector search top-k → expand k-hop neighborhood in graph → return context
- `src/rag/generator.py` — format context + chat history → LLM answer with citations
- `src/chat/session.py` + `prompts.py`
- **Verify**: Run a question through retriever → generator pipeline, check citations are real nodes in graph

### Stage 5 — Gradio GUI + Pipeline Orchestrator
- `src/pipeline.py` — `Pipeline.run()` async generator; yields `PipelineEvent` (paper processed, schema proposal, graph updated, done)
- `gui/app.py` — all 4 tabs; Setup → Schema → Pipeline uses `yield` to stream events; Chat tab calls `session.py`
- Session save/load: serialize NetworkX graph to .graphml + ChromaDB persist
- **Verify**: Full end-to-end run in GUI — enter topic, approve schema, watch graph build, ask question, verify cited answer

### Stage 6 — Polish + Documentation
- Add `docs/STRUCTURE.md` explaining every module
- Ensure all modules are well-commented (docstrings on every class and public method)
- Test with a second provider (OpenAI) to validate abstraction
- Test with a domain-different topic (e.g., "CRISPR gene editing" after "quantitative finance") to validate schema flexibility

---

## Dependencies (`requirements.txt`)

```
anthropic          # Claude API
openai             # OpenAI + compatible APIs (GLM4, Kimi via base_url)
arxiv              # arXiv paper search
networkx           # Knowledge graph
python-louvain     # Community detection
chromadb           # Vector store
pyvis              # Graph → interactive HTML
gradio             # GUI
pydantic           # Data validation
pyyaml             # Config file
python-dotenv      # .env loading
sentence-transformers  # Local embeddings (fallback when provider embed() not available)
```

---

## Verification (End-to-End)

1. `pip install -r requirements.txt`
2. Set API key in `.env`
3. `python gui/app.py` → Gradio opens at `localhost:7860`
4. Enter topic: *"deep reinforcement learning for algorithmic trading"*
5. Schema tab shows domain-specific types (e.g., `RL_Algorithm`, `Trading_Strategy`, `Market_Environment`)
6. Approve → Pipeline tab: watch graph build paper by paper, graph HTML updates each paper
7. After processing, trigger a schema evolution proposal (should appear if new entity types were flagged)
8. Chat tab: ask *"Which reinforcement learning algorithms have been applied to equity markets?"*
9. Answer should cite specific papers with arXiv IDs visible

---

## Notes for Grad School Presentation

- The schema generation and evolution step is itself a contribution — demonstrate it explicitly
- Community detection summaries should be visible in the GUI (collapsible panel)
- Save `.graphml` files per topic — can be inspected in Gephi for the write-up
- The `docs/STRUCTURE.md` is written last, after final architecture is stable
