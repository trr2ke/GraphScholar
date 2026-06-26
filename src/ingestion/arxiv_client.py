"""arXiv paper search and retrieval for GraphScholar."""
import arxiv

from .paper import Paper


class ArxivClient:
    """Thin wrapper around the arxiv library that returns Paper dataclasses."""

    def __init__(self, page_size: int = 100):
        # page_size controls the HTTP batch size internally used by the library
        self._client = arxiv.Client(page_size=page_size)

    def search(self, topic: str, max_results: int = 20) -> list[Paper]:
        """Search arXiv for papers matching a topic string.

        Args:
            topic: Free-text query forwarded to the arXiv search API.
            max_results: Maximum number of papers to return, sorted by relevance.

        Returns:
            List of Paper objects, most relevant first.
        """
        query = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers: list[Paper] = []
        for result in self._client.results(query):
            papers.append(self._to_paper(result))
        return papers

    def _to_paper(self, result: arxiv.Result) -> Paper:
        """Convert an arxiv.Result to a Paper dataclass."""
        # entry_id is a URL like https://arxiv.org/abs/2310.06825v3
        raw_id = result.entry_id.split("/")[-1]
        # Strip version suffix so IDs are stable across updates
        arxiv_id = raw_id.rsplit("v", 1)[0] if "v" in raw_id else raw_id

        return Paper(
            id=arxiv_id,
            title=result.title.strip(),
            abstract=result.summary.strip(),
            authors=[a.name for a in result.authors],
            url=f"https://arxiv.org/abs/{arxiv_id}",
            published=result.published.strftime("%Y-%m-%d"),
        )
