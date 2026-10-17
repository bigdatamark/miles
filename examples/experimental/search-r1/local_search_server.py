# [MODULE DOC] Client for the local retrieval server of the Search-R1 example, exposing one
# coroutine, local_search, with the same result shape as the Google backend so the generate module
# can use either. It posts a single query to the retrieval endpoint and returns a list of
# {"document": {"contents": text}} mappings.
"""
Local Search Server for Search-R1

This module provides a local search engine interface that mimics the google_search_server.py API.
It sends requests to a local retrieval server (e.g., running retrieval_server.py from Search-R1)
and formats the results to match the expected output format.

Usage:
    In your generate_with_search.py, replace:
        from google_search_server import google_search
    with:
        from local_search_server import local_search as google_search

    And update SEARCH_R1_CONFIGS:
        SEARCH_R1_CONFIGS = {
            "search_url": "http://127.0.0.1:8000/retrieve",  # URL of local retrieval server
            "topk": 3,
            ...
        }
"""

import aiohttp


async def local_search(
    search_url: str,
    query: str,
    top_k: int = 5,
    timeout: int = 60,
    proxy: str | None = None,
) -> list[dict]:
    # [FUNCTION DOC] Sends one query to the retrieval server and returns formatted results. Request
    # body: a one-item query list, the top-k count and scores disabled. Posts with the given total
    # timeout (seconds) and optional proxy; an HTTP error status raises inside a protected block. Any
    # exception while calling is caught, printed to standard output with the url, and an empty list
    # is returned (failures are silent to the caller). Takes the first query's result list from the
    # "result" entry (an empty list if absent) and, for each dictionary item, uses its "contents"
    # text directly as the document contents, or the placeholder '"No title."' line plus "No
    # snippet available." when the contents are empty; non-dictionary items are skipped.
    # KNOWN ISSUE: the docstring lists a snippet-only parameter that does not exist, the in-code
    # comments claim the document mapping is accessed first although items are read directly, and
    # the module usage text shows an outdated configuration layout; the proxy argument is honored
    # despite the note that it is unused.
    """
    Call local search engine server and format results to match google_search_server.py output.

    This function provides the same interface as google_search() from google_search_server.py,
    making it a drop-in replacement. The only difference is that instead of using an API key,
    it uses a search_url parameter.

    Args:
        search_url: URL of the local retrieval server (e.g., "http://127.0.0.1:8000/retrieve")
        query: Search query string
        top_k: Number of results to retrieve
        timeout: Request timeout in seconds (default: 60)
        proxy: Proxy URL if needed (not used for local retrieval, kept for API compatibility)
        snippet_only: If True, only return snippet (kept for API compatibility with google_search)

    Returns:
        List of dictionaries with format: [{"document": {"contents": '"<title>"\n<text>'}}]
        This matches the output format of google_search() from google_search_server.py
    """
    # Prepare request payload for local retrieval server
    payload = {
        "queries": [query],
        "topk": top_k,
        "return_scores": False,  # We don't need scores for compatibility with google_search_server
    }

    # Send async request to local retrieval server
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    session_kwargs = {}
    # Note: proxy parameter is kept for API compatibility but typically not needed for local server
    if proxy:
        session_kwargs["proxy"] = proxy

    try:
        async with aiohttp.ClientSession(**session_kwargs) as session:
            async with session.post(search_url, json=payload, timeout=timeout_obj) as resp:
                resp.raise_for_status()
                result = await resp.json()
    except Exception as e:
        print(f"Error calling local search engine at {search_url}: {e}")
        return []

    # Parse retrieval results
    # Format from retrieval_server.py: {"result": [[{"document": {"id": "...", "contents": "..."}}]]}
    retrieval_results = result.get("result", [[]])[0]
    # Format to match google_search_server.py output
    # Google format: [{"document": {"contents": '"<title>"\n<context>'}}]
    contexts = []

    for item in retrieval_results:
        # Extract contents from retrieval result
        # retrieval_server returns: {"document": {"id": "...", "contents": '"Title"\nText...'}}
        if isinstance(item, dict):
            # Access the document dict first, then get contents
            content = item.get("contents", "")

            if content:
                # The contents are already in the correct format: '"Title"\nText content...'
                # Just pass through as-is to match google_search format
                contexts.append({"document": {"contents": content}})
            else:
                # Empty content case - provide default values
                contexts.append({"document": {"contents": '"No title."\nNo snippet available.'}})

    # If no results found, return empty list (consistent with google_search_server.py)
    return contexts
