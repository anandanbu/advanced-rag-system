"""
tools/web_search.py
────────────────────
Optional web search tool for agents.
Uses DuckDuckGo (free, no API key needed) via the `duckduckgo-search` package.

Install: pip install duckduckgo-search

Used by agents when they need information not in the local documents.
Gracefully degrades if the package is not installed.
"""

from utils.logger import get_logger

logger = get_logger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.

    Args:
        query       : Search query string
        max_results : Number of results to return

    Returns:
        Formatted string with titles, URLs, and snippets.
        Returns an error message if search is unavailable.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return (
            "Web search unavailable. "
            "Install with: pip install duckduckgo-search"
        )

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('href', 'N/A')}\n"
                    f"Summary: {r.get('body', 'N/A')}"
                )

        if not results:
            return f"No web results found for: '{query}'"

        header = f"Web search results for: '{query}'\n{'─' * 40}\n"
        return header + "\n\n".join(results)

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search error: {str(e)}"


def news_search(query: str, max_results: int = 5) -> str:
    """
    Search for recent news articles using DuckDuckGo News.
    Useful for time-sensitive queries.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Web search unavailable. Install: pip install duckduckgo-search"

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"Source: {r.get('source', 'N/A')} | {r.get('date', 'N/A')}\n"
                    f"Summary: {r.get('body', 'N/A')}"
                )

        if not results:
            return f"No news results found for: '{query}'"

        return f"News results for '{query}':\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"News search error: {str(e)}"
