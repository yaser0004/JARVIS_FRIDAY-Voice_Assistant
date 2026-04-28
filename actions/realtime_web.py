from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, List, Tuple
from urllib.error import URLError
from urllib.parse import quote_plus, urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


REQUEST_TIMEOUT = 5
MAX_SOURCES_FOR_OVERVIEW = 7
MAX_NEWS_ITEMS = 3


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def _fetch_json(url: str) -> Dict[str, Any] | None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read().decode("utf-8", errors="ignore")
        return json.loads(raw)
    except (URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return None
    except Exception:
        return None


def _fetch_text(url: str) -> str | None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (URLError, TimeoutError, ValueError):
        return None
    except Exception:
        return None


def looks_like_research_query(text: str) -> bool:
    normalized = " ".join(str(text).lower().split())
    tokens = normalized.split()
    if len(tokens) < 4:
        return False

    small_talk = {
        "hi",
        "hello",
        "hey",
        "hey there",
        "how are you",
        "how are you doing",
        "how r you",
        "what's up",
        "whats up",
        "thanks",
        "thank you",
    }
    if normalized in small_talk:
        return False

    command_terms = [
        "open ",
        "launch ",
        "close ",
        "play ",
        "pause",
        "resume",
        "mute",
        "brightness",
        "volume",
        "shutdown",
        "restart",
        "sleep",
    ]
    if any(term in normalized for term in command_terms):
        return False

    explicit_web_markers = [
        "search web",
        "web search",
        "look up",
        "lookup",
        "find online",
        "from the web",
        "on the web",
        "internet",
    ]
    if any(marker in normalized for marker in explicit_web_markers):
        return True

    freshness_markers = [
        "latest",
        "news",
        "today",
        "current",
        "update",
        "updates",
        "recent",
        "breaking",
        "price",
        "weather",
        "score",
        "stock",
        "happening now",
    ]
    analysis_markers = ["what is", "who is", "when", "where", "why", "how", "explain", "compare", "difference"]

    has_freshness = any(marker in normalized for marker in freshness_markers)
    has_analysis = any(marker in normalized for marker in analysis_markers)
    if has_freshness and (has_analysis or "?" in str(text) or len(tokens) >= 4):
        return True

    # Keep broad general questions local unless user explicitly asks for web-backed/fresh info.
    return False


def _duckduckgo_summary(query: str) -> Tuple[str | None, str | None]:
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    url = f"https://api.duckduckgo.com/?{urlencode(params)}"
    payload = _fetch_json(url)
    if not payload:
        return None, None

    abstract = str(payload.get("AbstractText") or "").strip()
    abstract_url = str(payload.get("AbstractURL") or "").strip() or None

    if abstract:
        return abstract, abstract_url

    related = payload.get("RelatedTopics") or []
    if isinstance(related, list):
        for item in related:
            if isinstance(item, dict):
                text = str(item.get("Text") or "").strip()
                first_url = str(item.get("FirstURL") or "").strip() or None
                if text:
                    return text, first_url
                nested = item.get("Topics") or []
                if isinstance(nested, list):
                    for sub in nested:
                        if isinstance(sub, dict):
                            text = str(sub.get("Text") or "").strip()
                            first_url = str(sub.get("FirstURL") or "").strip() or None
                            if text:
                                return text, first_url

    return None, None


def _duckduckgo_related_summaries(query: str, limit: int = 3) -> List[Tuple[str, str | None]]:
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    url = f"https://api.duckduckgo.com/?{urlencode(params)}"
    payload = _fetch_json(url)
    if not payload:
        return []

    related = payload.get("RelatedTopics") or []
    if not isinstance(related, list):
        return []

    rows: List[Tuple[str, str | None]] = []

    def _collect_from_item(item: Dict[str, Any]) -> None:
        text = str(item.get("Text") or "").strip()
        first_url = str(item.get("FirstURL") or "").strip() or None
        if text:
            rows.append((text, first_url))

    for item in related:
        if len(rows) >= max(1, int(limit)):
            break
        if not isinstance(item, dict):
            continue

        if "Text" in item:
            _collect_from_item(item)
            continue

        nested = item.get("Topics") or []
        if isinstance(nested, list):
            for sub in nested:
                if len(rows) >= max(1, int(limit)):
                    break
                if isinstance(sub, dict):
                    _collect_from_item(sub)

    deduped: List[Tuple[str, str | None]] = []
    seen = set()
    for text, first_url in rows:
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((text, first_url))
    return deduped[: max(1, int(limit))]


def _wikipedia_summary(query: str) -> Tuple[str | None, str | None]:
    search_url = (
        "https://en.wikipedia.org/w/api.php?action=opensearch&limit=1&namespace=0&format=json&search="
        f"{quote_plus(query)}"
    )
    payload = _fetch_json(search_url)
    if not payload or not isinstance(payload, list) or len(payload) < 2:
        return None, None

    titles = payload[1]
    if not isinstance(titles, list) or not titles:
        return None, None

    page_title = str(titles[0]).strip()
    if not page_title:
        return None, None

    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(page_title)}"
    summary_payload = _fetch_json(summary_url)
    if not summary_payload:
        return None, None

    summary = str(summary_payload.get("extract") or "").strip()
    canonical = (
        str(summary_payload.get("content_urls", {}).get("desktop", {}).get("page") or "").strip() or None
    )
    return summary or None, canonical


def _wikidata_summary(query: str) -> Tuple[str | None, str | None]:
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": "en",
        "format": "json",
        "limit": "1",
    }
    url = f"https://www.wikidata.org/w/api.php?{urlencode(params)}"
    payload = _fetch_json(url)
    if not payload:
        return None, None

    rows = payload.get("search") or []
    if not isinstance(rows, list) or not rows:
        return None, None

    first = rows[0] if isinstance(rows[0], dict) else {}
    label = str(first.get("label") or "").strip()
    description = str(first.get("description") or "").strip()
    concept_url = str(first.get("concepturi") or "").strip() or None

    if label and description:
        return f"{label}: {description}", concept_url
    if label:
        return label, concept_url
    return None, concept_url


def _news_query(query: str) -> bool:
    normalized = " ".join(str(query).lower().split())
    triggers = {
        "latest",
        "news",
        "today",
        "current",
        "update",
        "updates",
        "recent",
        "headline",
        "headlines",
        "happening",
    }
    return any(token in normalized for token in triggers)


def _strip_html_tags(text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", no_tags).strip()


def _google_news_highlights(query: str) -> List[Tuple[str, str | None, str | None]]:
    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    url = f"https://news.google.com/rss/search?{urlencode(params)}"
    xml_payload = _fetch_text(url)
    if not xml_payload:
        return []

    try:
        root = ElementTree.fromstring(xml_payload)
    except ElementTree.ParseError:
        return []

    highlights: List[Tuple[str, str | None, str | None]] = []
    for item in root.findall("./channel/item"):
        title = str(item.findtext("title") or "").strip()
        description_raw = str(item.findtext("description") or "").strip()
        description = _strip_html_tags(html.unescape(description_raw)) if description_raw else None
        link = str(item.findtext("link") or "").strip() or None
        if not title:
            continue
        highlights.append((title, description or None, link))
        if len(highlights) >= MAX_NEWS_ITEMS:
            break
    return highlights


def _clean_snippet(text: str, max_chars: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"\[[0-9]+\]", "", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    for sentence in parts:
        if len(sentence.split()) >= 4:
            return sentence.strip()
    return str(text or "").strip()


def _source_display_name(source: Dict[str, str | None], index: int) -> str:
    provider = str(source.get("provider") or "").strip()
    if provider:
        return provider

    raw_url = str(source.get("url") or "").strip()
    if raw_url:
        normalized = raw_url if "://" in raw_url else f"https://{raw_url}"
        try:
            host = urlparse(normalized).netloc.strip().lower()
        except Exception:
            host = ""
        if host.startswith("www."):
            host = host[4:]
        if host:
            return host

    return f"Source {index}"


def _dedupe_sources(sources: List[Dict[str, str | None]]) -> List[Dict[str, str | None]]:
    seen = set()
    deduped: List[Dict[str, str | None]] = []

    for source in sources:
        snippet = str(source.get("snippet") or "").strip()
        if not snippet:
            continue
        signature = re.sub(r"[^a-z0-9]+", " ", snippet.lower()).strip()
        if not signature or signature in seen:
            continue
        seen.add(signature)
        deduped.append(source)

    return deduped


def _collect_sources(query: str) -> List[Dict[str, str | None]]:
    sources: List[Dict[str, str | None]] = []

    ddg_text, ddg_url = _duckduckgo_summary(query)
    if ddg_text:
        sources.append(
            {
                "provider": "DuckDuckGo",
                "snippet": _clean_snippet(ddg_text),
                "url": ddg_url,
            }
        )

    for related_text, related_url in _duckduckgo_related_summaries(query, limit=3):
        sources.append(
            {
                "provider": "DuckDuckGo related",
                "snippet": _clean_snippet(related_text),
                "url": related_url,
            }
        )

    wiki_text, wiki_url = _wikipedia_summary(query)
    if wiki_text:
        sources.append(
            {
                "provider": "Wikipedia",
                "snippet": _clean_snippet(wiki_text),
                "url": wiki_url,
            }
        )

    wikidata_text, wikidata_url = _wikidata_summary(query)
    if wikidata_text:
        sources.append(
            {
                "provider": "Wikidata",
                "snippet": _clean_snippet(wikidata_text),
                "url": wikidata_url,
            }
        )

    if _news_query(query):
        for title, description, link in _google_news_highlights(query):
            # Prefer article descriptions over titles to avoid headline-only summaries.
            snippet_source = description if description and len(description.split()) >= 6 else title
            sources.append(
                {
                    "provider": "Google News RSS",
                    "snippet": _clean_snippet(snippet_source, max_chars=260),
                    "url": link,
                }
            )

    return _dedupe_sources(sources)


def _consistency_score(chunks: List[str]) -> str:
    if len(chunks) < 2:
        return "limited"

    token_sets: List[set[str]] = []
    for text in chunks:
        words = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
        if words:
            token_sets.append(words)

    if len(token_sets) < 2:
        return "limited"

    overlaps: List[float] = []
    for index in range(len(token_sets)):
        for other_index in range(index + 1, len(token_sets)):
            overlap = token_sets[index] & token_sets[other_index]
            ratio = len(overlap) / max(1, min(len(token_sets[index]), len(token_sets[other_index])))
            overlaps.append(ratio)

    if not overlaps:
        return "limited"

    ratio = sum(overlaps) / len(overlaps)
    if ratio > 0.22:
        return "high"
    if ratio > 0.10:
        return "moderate"
    return "low"


def _looks_like_llm_error(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return True

    known_errors = [
        "missing model file",
        "initialization failed",
        "import failed",
        "worker error",
        "i/o is unavailable",
        "local llm",
        "traceback",
    ]
    return any(marker in normalized for marker in known_errors)


def _synthesize_with_llm(query: str, sources: List[Dict[str, str | None]], llm: Any | None) -> str | None:
    if llm is None:
        return None

    is_ready = getattr(llm, "is_ready", None)
    is_available = getattr(llm, "is_available", None)
    if callable(is_ready):
        try:
            ready = bool(is_ready())
        except Exception:
            return None

        if not ready:
            if callable(is_available):
                try:
                    if not bool(is_available()):
                        return None
                except Exception:
                    return None
            else:
                return None

    generate = getattr(llm, "generate", None)
    if not callable(generate):
        return None

    evidence_lines = []
    for index, source in enumerate(sources[:MAX_SOURCES_FOR_OVERVIEW], start=1):
        source_name = _source_display_name(source, index)
        snippet = str(source.get("snippet") or "").strip()
        if snippet:
            evidence_lines.append(f"[{index}] {source_name}: {snippet}")

    if not evidence_lines:
        return None

    prompt = (
        "Create a concise AI overview for the query using only the provided sources.\n"
        "Rules:\n"
        "- Write 3 to 5 sentences.\n"
        "- Stay faithful to the sources only.\n"
        "- Mention uncertainty when sources conflict or are weak.\n"
        "- Include bracket citations such as [1] and [2].\n\n"
        "- Do not output raw URLs; reference source names only.\n\n"
        f"Query: {query}\n\n"
        "Sources:\n"
        + "\n".join(evidence_lines)
    )

    try:
        candidate = str(generate(prompt, None)).strip()
    except Exception:
        return None

    if _looks_like_llm_error(candidate):
        return None
    return candidate


def _extractive_overview(sources: List[Dict[str, str | None]]) -> str:
    selected_sentences: List[str] = []
    seen = set()

    for source in sources[:MAX_SOURCES_FOR_OVERVIEW]:
        snippet = str(source.get("snippet") or "").strip()
        if not snippet:
            continue

        candidates = re.split(r"(?<=[.!?])\s+", snippet)
        picked_for_source = 0
        for raw_sentence in candidates:
            sentence = re.sub(r"\s+", " ", raw_sentence).strip(" \t\n\r:-")
            if not sentence or len(sentence.split()) < 5:
                continue
            if sentence[-1] not in ".!?":
                sentence = f"{sentence}."

            signature = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
            if not signature or signature in seen:
                continue

            seen.add(signature)
            selected_sentences.append(sentence)
            picked_for_source += 1

            if picked_for_source >= 2 or len(selected_sentences) >= 9:
                break

        if len(selected_sentences) >= 9:
            break

    if not selected_sentences:
        return "I found web signals, but not enough high-quality detail to compose an overview yet."

    if len(selected_sentences) <= 3:
        return " ".join(selected_sentences)

    paragraphs: List[str] = []
    for i in range(0, len(selected_sentences), 3):
        chunk = selected_sentences[i : i + 3]
        if chunk:
            paragraphs.append(" ".join(chunk))
        if len(paragraphs) >= 3:
            break

    return "\n\n".join(paragraphs)


def verified_answer(query: str, llm: Any | None = None) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return _response(False, "Please provide a query.")

    sources = _collect_sources(query)
    snippets = [str(item.get("snippet") or "") for item in sources if str(item.get("snippet") or "").strip()]
    if not snippets:
        return _response(False, "I could not retrieve trusted web snippets right now.")

    confidence = _consistency_score(snippets)
    overview = _extractive_overview(sources)
    synthesis_mode = "extractive"

    lines = ["AI web overview:", overview]

    if confidence == "high":
        lines.append("Cross-source consistency: high")
    elif confidence == "moderate":
        lines.append("Cross-source consistency: moderate")
    else:
        lines.append("Cross-source consistency: low, verify manually")

    source_names: List[str] = []
    seen_counts: Dict[str, int] = {}
    for index, source in enumerate(sources[:MAX_SOURCES_FOR_OVERVIEW], start=1):
        base_name = _source_display_name(source, index)
        normalized_name = base_name.strip().lower()
        if not normalized_name:
            continue
        seen_counts[normalized_name] = int(seen_counts.get(normalized_name, 0)) + 1
        if seen_counts[normalized_name] == 1:
            source_names.append(base_name)
        else:
            source_names.append(f"{base_name} ({seen_counts[normalized_name]})")

    if source_names:
        lines.append(f"Sources: {', '.join(source_names)}")
    else:
        lines.append("Sources: not available")

    source_urls = [str(item.get("url") or "").strip() for item in sources if str(item.get("url") or "").strip()]

    return _response(
        True,
        "\n".join(lines),
        {
            "mode": "verified_web_overview",
            "synthesis": synthesis_mode,
            "consistency": confidence,
            "sources": source_names,
            "source_urls": source_urls,
            "snippets": snippets,
            "providers": [str(item.get("provider") or "") for item in sources],
        },
    )
