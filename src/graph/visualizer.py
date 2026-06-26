"""NetworkX → PyVis interactive HTML visualizer for GraphScholar.

Renders the knowledge graph as a self-contained HTML file (or string) that
Gradio can embed in a gr.HTML component.  Nodes are colored by type and
sized by degree; edges carry a tooltip showing the relationship type.
"""
import networkx as nx
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Color palette — Paper and Author are fixed; domain types get the rest
# ---------------------------------------------------------------------------
_FIXED_COLORS = {
    "Paper":  "#4A90D9",   # blue
    "Author": "#9B59B6",   # purple
}
_PALETTE = [
    "#27AE60",  # green
    "#E74C3C",  # red
    "#F39C12",  # orange
    "#1ABC9C",  # teal
    "#E67E22",  # dark orange
    "#2980B9",  # dark blue
    "#8E44AD",  # dark purple
    "#16A085",  # dark teal
    "#D35400",  # burnt orange
    "#C0392B",  # dark red
]

# Physics options that produce a readable, well-spread layout
_PHYSICS_OPTIONS = """
{
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.3,
      "springLength": 160,
      "springConstant": 0.04,
      "damping": 0.09
    },
    "minVelocity": 0.75
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 150,
    "navigationButtons": true
  },
  "edges": {
    "smooth": {"type": "dynamic"},
    "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
    "font": {"size": 10, "align": "middle"}
  },
  "nodes": {
    "font": {"size": 12, "face": "arial"}
  }
}
"""


class GraphVisualizer:
    """Converts a NetworkX DiGraph to a PyVis interactive HTML string."""

    def __init__(self, height: str = "620px", width: str = "100%"):
        self.height = height
        self.width = width

    def render(self, graph: nx.DiGraph) -> str:
        """Render the graph and return a self-contained HTML string.

        Args:
            graph: The NetworkX DiGraph produced by GraphBuilder.

        Returns:
            HTML string suitable for gr.HTML or writing to a .html file.
        """
        net = Network(
            height=self.height,
            width=self.width,
            directed=True,
            notebook=False,
            bgcolor="#1a1a2e",   # dark background for contrast
            font_color="#e0e0e0",
        )
        net.set_options(_PHYSICS_OPTIONS)

        color_map = _build_color_map(graph)
        degree = dict(graph.degree())
        min_size, max_size = 12, 40

        # ── Nodes ────────────────────────────────────────────────────
        for nid, data in graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            label = data.get("label", nid)
            # Truncate long labels for display
            display = label if len(label) <= 30 else label[:28] + "…"
            color = color_map.get(node_type, "#95A5A6")

            # Size by degree, clamped to [min_size, max_size]
            deg = degree.get(nid, 1)
            size = min_size + (max_size - min_size) * min(deg / 10, 1.0)

            tooltip = f"<b>{node_type}</b><br>{label}"
            if "url" in data:
                tooltip += f"<br><a href='{data['url']}'>{data['url']}</a>"
            if "published" in data:
                tooltip += f"<br>Published: {data['published']}"

            net.add_node(
                nid,
                label=display,
                title=tooltip,
                color=color,
                size=size,
                group=node_type,
            )

        # ── Edges ────────────────────────────────────────────────────
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", "")
            weight = data.get("weight", 1)
            net.add_edge(
                u, v,
                title=edge_type,
                label=edge_type,
                width=1 + weight * 0.5,
                color={"opacity": 0.7},
            )

        return net.generate_html()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_color_map(graph: nx.DiGraph) -> dict[str, str]:
    """Assign a stable color to each node type in the graph."""
    domain_types = sorted(
        {data.get("type", "") for _, data in graph.nodes(data=True)}
        - set(_FIXED_COLORS)
    )
    result = dict(_FIXED_COLORS)
    for i, t in enumerate(domain_types):
        result[t] = _PALETTE[i % len(_PALETTE)]
    return result
