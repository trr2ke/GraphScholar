"""Gradio GUI for GraphScholar — four-tab interactive research assistant.

Tabs
----
1. Setup    — topic, provider, model, API key, n_papers → generate schema
2. Schema   — review / edit proposed schema → approve & begin ingestion
3. Pipeline — live progress, graph HTML, schema evolution proposals
4. Chat     — conversational Q&A with citation-grounded answers
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


_CONFIG = _load_config()

# ---------------------------------------------------------------------------
# Schema ↔ Dataframe helpers
# ---------------------------------------------------------------------------

def _schema_to_tables(schema) -> tuple[list, list]:
    """Convert a GraphSchema to (node_rows, edge_rows) for gr.Dataframe."""
    node_rows = [
        [nt.name, nt.description, ", ".join(nt.properties)]
        for nt in schema.node_types
    ]
    edge_rows = [
        [et.name, et.description, et.from_type, et.to_type]
        for et in schema.edge_types
    ]
    return node_rows, edge_rows


def _df_to_rows(df_or_list) -> list[list]:
    """Normalise a Gradio Dataframe value (pandas or list) to list-of-lists."""
    if df_or_list is None:
        return []
    if hasattr(df_or_list, "fillna"):          # pandas DataFrame
        return df_or_list.fillna("").values.tolist()
    return list(df_or_list)


def _tables_to_schema(node_rows, edge_rows, topic: str, version: int = 1):
    """Reconstruct a GraphSchema from editable Dataframe rows."""
    from src.extraction.schema import EdgeType, GraphSchema, NodeType

    node_types = []
    for row in _df_to_rows(node_rows):
        name = str(row[0]).strip() if row and row[0] else ""
        if not name:
            continue
        props_raw = str(row[2]).strip() if len(row) > 2 and row[2] else ""
        props = [p.strip() for p in props_raw.split(",") if p.strip()]
        node_types.append(NodeType(
            name=name,
            description=str(row[1] or "").strip(),
            properties=props,
        ))

    edge_types = []
    for row in _df_to_rows(edge_rows):
        name = str(row[0]).strip() if row and row[0] else ""
        if not name:
            continue
        edge_types.append(EdgeType(
            name=name,
            description=str(row[1] or "").strip(),
            from_type=str(row[2] or "").strip(),
            to_type=str(row[3] or "").strip(),
        ))

    return GraphSchema(topic=topic, node_types=node_types, edge_types=edge_types, version=version)


# ---------------------------------------------------------------------------
# Proposal display helper
# ---------------------------------------------------------------------------

def _proposals_to_markdown(proposals: list[dict]) -> str:
    """Render pending schema proposals as Markdown cards."""
    if not proposals:
        return "_No pending proposals._"
    lines = []
    for p in proposals:
        if p.get("kind") == "node":
            props = ", ".join(p.get("properties", [])) or "—"
            lines.append(
                f"**Node type: `{p['name']}`**  \n"
                f"{p.get('description', '')}  \n"
                f"*Properties:* {props}"
            )
        else:
            lines.append(
                f"**Edge type: `{p['name']}`**  \n"
                f"{p.get('description', '')}  \n"
                f"`{p.get('from_type', '?')}` → `{p.get('to_type', '?')}`"
            )
    return "\n\n---\n\n".join(lines)


# ---------------------------------------------------------------------------
# Sources panel helper
# ---------------------------------------------------------------------------

def _sources_markdown(pipe) -> str:
    """Build the sources panel from the pipeline's last retrieval result."""
    retrieval = getattr(pipe, "last_retrieval", None)
    if retrieval is None:
        return "### Sources\n_Ask a question to see retrieved papers and graph entities._"

    lines = ["### Retrieved Sources\n"]
    for p in retrieval.papers:
        lines.append(
            f"**[{p['id']}]** {p['title']}  \n"
            f"_{p['authors']} ({p['published']})_"
        )
    if retrieval.graph_nodes:
        lines.append("\n**Related graph entities:**")
        for n in retrieval.graph_nodes[:12]:
            lines.append(f"- {n.get('type', '?')}: {n.get('label', '?')}")
    if retrieval.community_summaries:
        lines.append("\n**Thematic clusters:**")
        for c in retrieval.community_summaries:
            lines.append(f"- Cluster {c['id']}: {c['summary'][:160]}…")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_EMPTY_GRAPH_HTML = (
    "<div style='height:620px;display:flex;align-items:center;"
    "justify-content:center;background:#1a1a2e;color:#a0a0c0;"
    "border-radius:8px;font-family:arial;font-size:14px'>"
    "Graph will appear here after ingestion starts.</div>"
)


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""

    with gr.Blocks(title="GraphScholar", theme=gr.themes.Soft()) as app:

        gr.Markdown(
            "# GraphScholar\n"
            "Knowledge graph pipeline for scientific literature — "
            "ingest arXiv papers, extract entities, ask questions."
        )

        # ── Shared state ─────────────────────────────────────────────
        pipeline_state  = gr.State(None)   # Pipeline instance
        topic_state     = gr.State("")     # current topic string
        proposals_state = gr.State([])     # accumulated schema proposals

        with gr.Tabs():

            # ── Tab 1: Setup ─────────────────────────────────────────
            with gr.Tab("Setup"):
                gr.Markdown("### Step 1 — Configure your session")
                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="e.g., deep reinforcement learning for algorithmic trading",
                )
                with gr.Row():
                    provider_dd = gr.Dropdown(
                        choices=["claude", "openai", "kimi", "glm4"],
                        value=_CONFIG.get("provider", "claude"),
                        label="LLM Provider",
                    )
                    model_input = gr.Textbox(
                        label="Model Name",
                        value=_CONFIG.get("model", "claude-opus-4-8"),
                    )
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="Leave blank to use .env / environment variable",
                    )
                    base_url_input = gr.Textbox(
                        label="API Base URL",
                        placeholder="Kimi / GLM4 only — leave blank for Claude / OpenAI",
                    )
                n_papers_slider = gr.Slider(
                    minimum=3, maximum=30,
                    value=_CONFIG.get("n_papers", 10),
                    step=1,
                    label="Number of Papers",
                )
                generate_schema_btn = gr.Button("Generate Schema →", variant="primary")
                setup_status = gr.Markdown("")

            # ── Tab 2: Schema ─────────────────────────────────────────
            with gr.Tab("Schema"):
                gr.Markdown(
                    "### Step 2 — Review and edit the proposed schema\n"
                    "Add, remove, or rename types, then approve to begin ingestion."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Node Types**")
                        node_table = gr.Dataframe(
                            headers=["Name", "Description", "Properties (comma-sep)"],
                            datatype=["str", "str", "str"],
                            row_count=(5, "dynamic"),
                            col_count=(3, "fixed"),
                            interactive=True,
                            label="Node Types",
                        )
                    with gr.Column():
                        gr.Markdown("**Edge Types**")
                        edge_table = gr.Dataframe(
                            headers=["Name", "Description", "From Type", "To Type"],
                            datatype=["str", "str", "str", "str"],
                            row_count=(5, "dynamic"),
                            col_count=(4, "fixed"),
                            interactive=True,
                            label="Edge Types",
                        )
                approve_btn = gr.Button("Approve & Ingest →", variant="primary")
                schema_status = gr.Markdown("")

            # ── Tab 3: Pipeline ───────────────────────────────────────
            with gr.Tab("Pipeline"):
                pipeline_status = gr.Markdown(
                    "### Pipeline Status\nWaiting to start…"
                )
                graph_html = gr.HTML(_EMPTY_GRAPH_HTML)
                with gr.Accordion("Schema Evolution Proposals", open=True):
                    proposals_md = gr.Markdown("_No proposals yet._")
                    with gr.Row():
                        accept_btn = gr.Button("Accept All Proposals", variant="primary")
                        reject_btn = gr.Button("Reject All", variant="secondary")
                    evolution_result = gr.Markdown("")
                pipeline_log = gr.Textbox(
                    label="Processing Log", lines=8, interactive=False
                )

            # ── Tab 4: Chat ───────────────────────────────────────────
            with gr.Tab("Chat"):
                gr.Markdown("### Step 4 — Ask questions about the literature")
                chatbot = gr.Chatbot(
                    label="Research Assistant",
                    height=440,
                    type="messages",
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Your question",
                        placeholder="Which RL algorithms have been applied to equity markets?",
                        scale=5,
                    )
                    chat_btn = gr.Button("Ask", variant="primary", scale=1)
                sources_md = gr.Markdown(
                    "### Sources\nAsk a question to see retrieved papers and graph entities."
                )

        # ── Event: generate schema ────────────────────────────────────

        def on_generate_schema(topic, provider_name, model_name, api_key, base_url, n_papers):
            """Phase 1: build a Pipeline instance and propose the initial schema."""
            if not topic.strip():
                return gr.update(), gr.update(), None, "", "⚠ Please enter a research topic."
            try:
                from src.api.factory import get_provider
                from src.pipeline import Pipeline

                cfg = dict(_CONFIG)
                cfg["model"] = model_name.strip()
                if base_url.strip():
                    cfg["base_url"] = base_url.strip()
                if api_key.strip():
                    env_key = (
                        "ANTHROPIC_API_KEY" if provider_name == "claude"
                        else "OPENAI_API_KEY"
                    )
                    os.environ[env_key] = api_key.strip()

                pipe = Pipeline(
                    get_provider(provider_name, cfg),
                    persist_dir="data/vectors",
                    similarity_threshold=float(cfg.get("similarity_threshold", 0.85)),
                )
                schema = pipe.generate_schema(topic.strip())
                node_rows, edge_rows = _schema_to_tables(schema)

                msg = (
                    f"✅ Schema generated for **{topic.strip()}** — "
                    f"{len(schema.node_types)} node types, "
                    f"{len(schema.edge_types)} edge types. "
                    "Switch to the **Schema** tab to review."
                )
                return node_rows, edge_rows, pipe, topic.strip(), msg

            except Exception as exc:
                return gr.update(), gr.update(), None, "", f"❌ Error: {exc}"

        generate_schema_btn.click(
            on_generate_schema,
            inputs=[topic_input, provider_dd, model_input, api_key_input, base_url_input, n_papers_slider],
            outputs=[node_table, edge_table, pipeline_state, topic_state, setup_status],
        )

        # ── Event: approve schema & begin ingestion (streaming) ───────

        def on_ingest(pipe, topic, node_rows, edge_rows, n_papers):
            """Phase 2: rebuild schema from GUI tables, then stream ingestion events."""
            if pipe is None:
                yield (
                    "❌ Generate a schema first (Setup tab).",
                    gr.update(), gr.update(), gr.update(), [], "",
                )
                return

            # Rebuild schema from the (possibly edited) Dataframe tables
            try:
                schema_version = pipe._schema.version if pipe._schema else 1
                schema = _tables_to_schema(node_rows, edge_rows, topic, schema_version)
                pipe.set_schema(schema)
            except Exception as exc:
                yield (
                    f"❌ Schema parse error: {exc}",
                    gr.update(), gr.update(), gr.update(), [], "",
                )
                return

            log_lines: list[str] = []
            all_proposals: list[dict] = []

            for event in pipe.ingest(topic, int(n_papers)):

                if event.type == "paper_processed":
                    p = event.payload
                    log_lines.append(
                        f"[{p['paper_index']}/{p['total_papers']}] {p['paper_title'][:80]}"
                    )
                    status = (
                        f"### Processing: {p['paper_index']} / {p['total_papers']} papers\n"
                        f"Graph: **{p['node_count']} nodes**, **{p['edge_count']} edges**"
                    )
                    yield (
                        status,
                        p.get("graph_html", gr.update()),
                        _proposals_to_markdown(all_proposals),
                        "\n".join(log_lines[-25:]),
                        all_proposals,
                        "",
                    )

                elif event.type == "schema_proposal":
                    p = event.payload
                    for n in p.get("proposed_nodes", []):
                        all_proposals.append({**n, "kind": "node"})
                    for e in p.get("proposed_edges", []):
                        all_proposals.append({**e, "kind": "edge"})
                    log_lines.append(
                        f"⚡ Schema proposal: "
                        f"{len(p.get('proposed_nodes', []))} node type(s), "
                        f"{len(p.get('proposed_edges', []))} edge type(s)"
                    )
                    yield (
                        gr.update(),
                        gr.update(),
                        _proposals_to_markdown(all_proposals),
                        "\n".join(log_lines[-25:]),
                        all_proposals,
                        "",
                    )

                elif event.type == "community_detected":
                    p = event.payload
                    n = p["n_communities"]
                    log_lines.append(
                        f"🔵 Communities: {n} cluster{'s' if n != 1 else ''} detected"
                    )
                    status = (
                        f"### Community detection complete — {n} cluster{'s' if n != 1 else ''}\n"
                        f"Graph: **{pipe.graph.number_of_nodes()} nodes**, "
                        f"**{pipe.graph.number_of_edges()} edges**"
                    )
                    yield (
                        status,
                        p.get("graph_html", gr.update()),
                        _proposals_to_markdown(all_proposals),
                        "\n".join(log_lines[-25:]),
                        all_proposals,
                        "",
                    )

                elif event.type == "done":
                    p = event.payload
                    log_lines.append(
                        f"✅ Done — {p['node_count']} nodes, {p['edge_count']} edges"
                    )
                    status = (
                        f"### ✅ Pipeline complete!\n"
                        f"**{p['node_count']} nodes**, **{p['edge_count']} edges**  ·  "
                        f"Switch to the **Chat** tab to ask questions."
                    )
                    yield (
                        status,
                        gr.update(),
                        _proposals_to_markdown(all_proposals),
                        "\n".join(log_lines[-25:]),
                        all_proposals,
                        "",
                    )

                elif event.type == "error":
                    log_lines.append(f"⚠ {event.payload.get('message', 'Unknown error')}")
                    yield (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "\n".join(log_lines[-25:]),
                        all_proposals,
                        "",
                    )

        approve_btn.click(
            on_ingest,
            inputs=[pipeline_state, topic_state, node_table, edge_table, n_papers_slider],
            outputs=[
                pipeline_status, graph_html, proposals_md,
                pipeline_log, proposals_state, evolution_result,
            ],
        )

        # ── Event: accept / reject schema proposals ───────────────────

        def on_accept_proposals(pipe, proposals):
            """Apply all pending proposals to the active schema."""
            if pipe is None or not proposals:
                return proposals, "⚠ No pipeline or proposals to accept."
            accepted_nodes = [p for p in proposals if p.get("kind") == "node"]
            accepted_edges = [p for p in proposals if p.get("kind") == "edge"]
            pipe.apply_evolution(accepted_nodes, accepted_edges)
            return [], (
                f"✅ Accepted {len(accepted_nodes)} node type(s) and "
                f"{len(accepted_edges)} edge type(s).  "
                f"Schema is now v{pipe._schema.version}."
            )

        accept_btn.click(
            on_accept_proposals,
            inputs=[pipeline_state, proposals_state],
            outputs=[proposals_state, evolution_result],
        )

        def on_reject_proposals(proposals):
            """Dismiss all pending proposals without changing the schema."""
            count = len(proposals)
            return [], f"Rejected {count} proposal(s)."

        reject_btn.click(
            on_reject_proposals,
            inputs=[proposals_state],
            outputs=[proposals_state, evolution_result],
        )

        # ── Event: chat ───────────────────────────────────────────────

        def on_chat(query, history, pipe):
            """Run the RAG pipeline and return an answer with citations."""
            history = history or []
            if not query.strip():
                return history, gr.update()
            if pipe is None:
                history = history + [
                    {"role": "user", "content": query},
                    {"role": "assistant",
                     "content": "⚠ Please complete ingestion first "
                                "(Setup → Schema → Pipeline tabs)."},
                ]
                return history, gr.update()

            try:
                answer = pipe.answer(query.strip())
            except Exception as exc:
                answer = f"❌ Error generating answer: {exc}"

            history = history + [
                {"role": "user",      "content": query},
                {"role": "assistant", "content": answer},
            ]
            return history, _sources_markdown(pipe)

        chat_btn.click(
            on_chat,
            inputs=[chat_input, chatbot, pipeline_state],
            outputs=[chatbot, sources_md],
        ).then(lambda: "", outputs=[chat_input])

        chat_input.submit(
            on_chat,
            inputs=[chat_input, chatbot, pipeline_state],
            outputs=[chatbot, sources_md],
        ).then(lambda: "", outputs=[chat_input])

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    create_app().launch(share=False)
