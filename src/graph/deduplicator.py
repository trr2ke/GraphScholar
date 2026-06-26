"""Embedding-based node deduplication for GraphScholar.

When the same concept appears under different surface forms across papers
("LSTM", "Long Short-Term Memory", "LSTM model"), the deduplicator maps
all variants to a single canonical label so the graph stays clean.

Algorithm per paper batch:
  1. Group new (type, label) pairs by node type.
  2. Batch-embed all new labels for a given type in one provider call.
  3. Compute cosine similarity against every existing label of that type.
  4. If max similarity >= threshold → merge into the existing canonical label.
  5. Otherwise → add the new label to the registry as a new canonical node.
"""
import numpy as np

from ..api.base import BaseLLMProvider


class NodeDeduplicator:
    """Resolves new node labels to canonical labels using embedding similarity."""

    def __init__(self, provider: BaseLLMProvider, threshold: float = 0.85):
        """
        Args:
            provider: LLM provider used for embed() calls.
            threshold: Cosine similarity above which two labels are considered
                       the same entity and merged into the existing one.
        """
        self.provider = provider
        self.threshold = threshold
        # {node_type: [(canonical_label, embedding_vector), ...]}
        self._registry: dict[str, list[tuple[str, list[float]]]] = {}

    def resolve_batch(
        self, candidates: list[tuple[str, str]]
    ) -> dict[tuple[str, str], str]:
        """Resolve a batch of (type, label) pairs to canonical labels.

        Processes all candidates from one paper at once to minimise embed() calls.

        Args:
            candidates: List of (node_type, label) pairs from one extraction pass.

        Returns:
            Mapping {(node_type, original_label): canonical_label}.
            When two labels resolve to the same canonical, the older one wins.
        """
        # Group new labels by type so we can batch-embed per type
        by_type: dict[str, list[str]] = {}
        for node_type, label in candidates:
            by_type.setdefault(node_type, []).append(label)

        result: dict[tuple[str, str], str] = {}

        for node_type, new_labels in by_type.items():
            existing = self._registry.get(node_type, [])
            existing_labels = [label for label, _ in existing]
            existing_vecs = [vec for _, vec in existing]

            # Embed all new labels for this type in one call
            new_vecs = self.provider.embed(new_labels)

            for label, vec in zip(new_labels, new_vecs):
                canonical = label

                if existing_vecs:
                    sims = [_cosine(vec, ev) for ev in existing_vecs]
                    best_idx = int(np.argmax(sims))
                    if sims[best_idx] >= self.threshold:
                        canonical = existing_labels[best_idx]

                # If not merged into an existing node, register as new canonical
                if canonical == label:
                    self._registry.setdefault(node_type, []).append((label, vec))
                    existing_labels.append(label)
                    existing_vecs.append(vec)

                result[(node_type, label)] = canonical

        return result

    def known_labels(self, node_type: str) -> list[str]:
        """Return all canonical labels registered for a given node type."""
        return [label for label, _ in self._registry.get(node_type, [])]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0
