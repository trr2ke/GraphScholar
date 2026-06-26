"""Knowledge graph schema dataclasses for GraphScholar.

A GraphSchema is proposed by the LLM at topic-entry time (Option C:
topic-seeded schema that evolves incrementally as papers are processed).
Each evolution increments `version` so callers can detect schema changes.
"""
import json
from dataclasses import asdict, dataclass, field


@dataclass
class NodeType:
    """A type of entity node in the knowledge graph."""

    name: str             # PascalCase, e.g. "AI_Technique"
    description: str      # shown in GUI and injected into extraction prompts
    properties: list[str] = field(default_factory=list)  # attribute names, e.g. ["name"]


@dataclass
class EdgeType:
    """A typed directional relationship between two node types."""

    name: str             # SCREAMING_SNAKE_CASE, e.g. "APPLIED_TO"
    description: str
    from_type: str        # must match a NodeType.name in the same schema
    to_type: str          # must match a NodeType.name in the same schema


@dataclass
class GraphSchema:
    """Complete schema for one research topic's knowledge graph.

    Always contains at least Paper and Author node types (structural anchors).
    Domain-specific types are proposed by the LLM and may grow via evolution.
    """

    topic: str
    node_types: list[NodeType]
    edge_types: list[EdgeType]
    version: int = 1      # increments on each user-approved schema evolution

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "GraphSchema":
        return cls(
            topic=data["topic"],
            node_types=[NodeType(**n) for n in data["node_types"]],
            edge_types=[EdgeType(**e) for e in data["edge_types"]],
            version=data.get("version", 1),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "GraphSchema":
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Convenience helpers for extraction prompts
    # ------------------------------------------------------------------

    def node_names(self) -> list[str]:
        return [n.name for n in self.node_types]

    def edge_names(self) -> list[str]:
        return [e.name for e in self.edge_types]

    def summary(self) -> str:
        """One-line human-readable description for logging / GUI."""
        return (
            f"v{self.version} — "
            f"{len(self.node_types)} node types: {', '.join(self.node_names())} | "
            f"{len(self.edge_types)} edge types: {', '.join(self.edge_names())}"
        )
