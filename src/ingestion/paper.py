"""Paper dataclass — the unit of ingestion from arXiv."""
from dataclasses import dataclass


@dataclass
class Paper:
    """A single arXiv paper with the fields needed for extraction and display."""

    id: str           # arXiv ID without version, e.g. "2310.06825"
    title: str
    abstract: str
    authors: list[str]
    url: str          # canonical abstract URL, e.g. "https://arxiv.org/abs/2310.06825"
    published: str    # ISO 8601 date string, e.g. "2023-10-10"

    def short_authors(self, max_names: int = 3) -> str:
        """Return a readable author string, truncated for display."""
        names = self.authors[:max_names]
        suffix = " et al." if len(self.authors) > max_names else ""
        return ", ".join(names) + suffix
