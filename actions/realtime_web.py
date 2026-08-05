from __future__ import annotations

import html
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus, unquote, urlencode, urlparse
from xml.etree import ElementTree

from actions.common import fetch_json, fetch_text, response as _response
from core.session_logging import trace_event, trace_exception


REQUEST_TIMEOUT = 5
# Whole-fan-out budget for _collect_sources; providers run concurrently.
COLLECT_DEADLINE = 8.0
MAX_SOURCES_FOR_OVERVIEW = 7
MAX_NEWS_ITEMS = 3

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a factual summarization engine. Follow the user's formatting rules exactly. "
    "Output only the summary text - no preamble, no meta-commentary, no repeating the rules."
)


_fetch_json = partial(fetch_json, timeout=REQUEST_TIMEOUT)
_fetch_text = partial(fetch_text, timeout=REQUEST_TIMEOUT)
_trace = partial(trace_event, "backend.realtime_web")
_trace_exc = partial(trace_exception, "backend.realtime_web")


_SMALL_TALK_SET = {
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

_COMMAND_TERMS = [
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

# Tier 1: explicit user instruction to go fetch something from the web. Wins
# over every other signal, including conceptual phrasing in the same sentence.
# Note: matches are only trusted near the start of the utterance (see
# _explicit_match_near_start) - a long sentence that merely mentions "web
# search" while describing/complaining about the feature ("...both of which
# were asked with web search on...") isn't a command to search anything.
_EXPLICIT_WEB_MARKERS = [
    "search web",
    "web search",
    "search the web",
    "look up",
    "lookup",
    "find online",
    "from the web",
    "on the web",
    "search online",
    "google it",
]

_PERSONAL_TASK_MARKERS = [
    "should i",
    "help me",
    "write me",
    "draft",
    "translate",
    "summarize this",
    "resume",
    "cv",
    "cover letter",
    "essay",
]

# Tier 2: genuine live/time-sensitive data needs. Domain terms alone are
# sufficient; weak temporal words only count when paired with a domain term.
_LIVE_DOMAIN_MARKERS = [
    "news",
    "headline",
    "headlines",
    "breaking",
    "weather",
    "forecast",
    "temperature",
    "stock price",
    "share price",
    "market price",
    "exchange rate",
    "score",
    "fixture",
    "who won",
    "election result",
    "election results",
    "latest version",
    "new version",
    "release of",
    "just released",
    "price of",
    "release date",
    "set to release",
    "coming out",
    "due for release",
    "launch date",
    "when is it coming",
]

_WEAK_TEMPORAL_MARKERS = ["latest", "today", "current", "update", "updates", "recent"]

# Regex fallbacks for explicit-search phrasings that a literal substring list
# misses because of a pronoun in between ("look IT up", "search THAT for me").
_EXPLICIT_WEB_PATTERNS = [
    re.compile(r"\blook\s+(it|that|this|him|her|them)?\s*up\b"),
    re.compile(r"\bsearch\s+(it|that|this|for\s+(it|that|this))\b"),
]

# Markers that always signal "give me an explanation," regardless of any
# temporal word present. Checked before domain terms so a phrase like "explain
# how stock prices are determined" doesn't false-trigger on "stock price".
_HARD_CONCEPTUAL_MARKERS = [
    "explain",
    "how does",
    "how do i",
    "how to",
    "why",
    "compare",
    "difference between",
    "meaning of",
    "pros and cons",
    "define",
    "definition of",
]

# Ambiguous openers: "what is recursion" (conceptual) vs "what is the latest
# Galaxy Z Fold phone" (factual/current). Checked LAST, only once no live-data
# signal (domain term or weak-temporal + content word) has already resolved it.
_SOFT_CONCEPTUAL_MARKERS = ["what is", "what are", "who is", "who was"]

# Always locally answerable: the system prompt already injects the real
# current date/time, so "current"/"latest" here isn't a genuine web-data need
# and would otherwise false-trigger via the weak-temporal + content-word test.
_LOCALLY_ANSWERABLE_MARKERS = [
    "current time",
    "what time is it",
    "what's the time",
    "whats the time",
    "current date",
    "what's the date",
    "whats the date",
    "what day is it",
    "today's date",
    "todays date",
]

_QUERY_STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "which", "does", "did", "do",
    "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "for", "to",
    "and", "or", "tell", "me", "about", "explain", "define",
}

# Extra stopwords for the content-word test only: common contractions and
# meta/command verbs that regex-tokenize as standalone 3+ char words but carry
# no topical content of their own (e.g. "look it up" shouldn't count "look").
_CONTENT_TEST_STOPWORDS = _QUERY_STOPWORDS | {
    "whats", "hows", "thats", "theres", "youre", "theyre", "its", "cant",
    "dont", "doesnt", "didnt", "wont", "just", "please", "know", "right", "now",
    "look", "search", "find", "check", "get", "give", "yeah", "yea", "cuz",
    "far", "you", "your", "this", "that", "these", "those", "not", "have",
    "has", "had", "can", "could", "would", "will", "shall", "should", "with",
}


def _significant_tokens(text: str, stopwords: set[str] | None = None) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z]{3,}", str(text or "").lower()))
    if stopwords:
        tokens -= stopwords
    return tokens


def _has_content_word(normalized: str, exclude: List[str]) -> bool:
    """True if the query has a substantive token beyond stopwords/temporal words.

    Case-insensitive by design - unlike a capitalized-token check, this fires
    for real-world lowercase input ("whats the latest iphone?"), which is the
    norm for typed/voice queries, not the exception.
    """
    tokens = _significant_tokens(normalized, _CONTENT_TEST_STOPWORDS)
    tokens -= {word for word in exclude if len(word) >= 3}
    return bool(tokens)


def has_topic_content(text: str) -> bool:
    """True if text carries a real topic beyond stopwords/interrogatives/temporal words.

    Used by the router to decide whether a short follow-up like "look it up"
    or "when was it released" needs its search topic spliced in from recent
    conversation context before being sent to verified_answer.
    """
    normalized = " ".join(str(text).lower().split())
    return _has_content_word(normalized, _WEAK_TEMPORAL_MARKERS)


# Explicit markers are only trusted this close to the start of the utterance.
# A genuine command front-loads it ("search the web for X", "look that up");
# a long sentence that merely references "web search" deep in a subordinate
# clause while describing/complaining about the feature is not a command.
_EXPLICIT_MAX_WORD_POSITION = 12


def _explicit_match_near_start(normalized: str) -> bool:
    for marker in _EXPLICIT_WEB_MARKERS:
        idx = normalized.find(marker)
        if idx >= 0 and len(normalized[:idx].split()) < _EXPLICIT_MAX_WORD_POSITION:
            return True
    for pattern in _EXPLICIT_WEB_PATTERNS:
        match = pattern.search(normalized)
        if match and len(normalized[: match.start()].split()) < _EXPLICIT_MAX_WORD_POSITION:
            return True
    return False


def classify_web_need(text: str) -> str:
    """Classify whether text needs a real web lookup: "explicit" | "live" | "none"."""
    normalized = " ".join(str(text).lower().split())
    if not normalized:
        return "none"

    if _explicit_match_near_start(normalized):
        return "explicit"

    if any(marker in normalized for marker in _LOCALLY_ANSWERABLE_MARKERS):
        return "none"

    # Hard-conceptual ("explain", "how do i", ...) always means "give me an
    # explanation," even if a temporal/domain word is also present.
    if any(marker in normalized for marker in _HARD_CONCEPTUAL_MARKERS):
        return "none"

    if any(marker in normalized for marker in _PERSONAL_TASK_MARKERS):
        return "none"

    has_domain_term = any(marker in normalized for marker in _LIVE_DOMAIN_MARKERS)
    if has_domain_term:
        return "live"

    has_weak_temporal = any(marker in normalized for marker in _WEAK_TEMPORAL_MARKERS)
    if has_weak_temporal and _has_content_word(normalized, _WEAK_TEMPORAL_MARKERS):
        return "live"

    # Only reached once no live-data signal matched: bare "what is"/"who is"
    # openers with nothing else going on are conceptual/historical, not live.
    if any(marker in normalized for marker in _SOFT_CONCEPTUAL_MARKERS):
        return "none"

    return "none"


def looks_like_research_query(text: str) -> bool:
    normalized = " ".join(str(text).lower().split())
    tokens = normalized.split()
    if len(tokens) < 4:
        return False

    if normalized in _SMALL_TALK_SET:
        return False

    if any(term in normalized for term in _COMMAND_TERMS):
        return False

    return classify_web_need(text) != "none"


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


_DDG_LITE_RESULT_RE = re.compile(
    r"<a rel=\"nofollow\" href=\"//duckduckgo\.com/l/\?uddg=(?P<url>[^&\"]+)[^\"]*\" class='result-link'>"
    r"(?P<title>.*?)</a>.*?result-snippet'>(?P<snippet>.*?)</td>",
    re.DOTALL,
)


def _ddg_lite_search(query: str, limit: int = 3) -> List[Dict[str, str | None]]:
    """Scrape DuckDuckGo's lite HTML results as a general organic-search fallback.

    The Instant Answer API (`_duckduckgo_summary`) only fires for a narrow set of
    entity-style queries and returns nothing for most everyday factual questions
    (e.g. "what is the capital of france"). Lite HTML results cover those gaps.
    """
    url = f"https://lite.duckduckgo.com/lite/?{urlencode({'q': query})}"
    payload = _fetch_text(url)
    if not payload:
        return []

    rows: List[Dict[str, str | None]] = []
    for match in _DDG_LITE_RESULT_RE.finditer(payload):
        title = _strip_html_tags(html.unescape(match.group("title")))
        snippet = _strip_html_tags(html.unescape(match.group("snippet")))
        raw_url = match.group("url")
        try:
            page_url = unquote(raw_url) or None
        except Exception:
            page_url = None
        if not snippet:
            continue
        rows.append(
            {
                "provider": "DuckDuckGo Search",
                "snippet": _clean_snippet(f"{title}: {snippet}" if title else snippet),
                "url": page_url,
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def _best_matching_row(
    rows: List[Dict[str, Any]], query_tokens: set[str], text_fields: Tuple[str, ...]
) -> Dict[str, Any] | None:
    """Pick the row whose text fields overlap the query best; drop on zero overlap.

    A relevance gate against ambiguous-name providers (Wikipedia/Wikidata) that
    otherwise blindly trust whichever candidate the search API ranked first.
    """
    candidates = [row for row in rows if isinstance(row, dict)]
    if not candidates:
        return None
    if not query_tokens:
        # No significant tokens to compare (very short/stopword-only query) - trust the top hit.
        return candidates[0]

    best_row: Dict[str, Any] | None = None
    best_score = 0
    for row in candidates:
        candidate_text = " ".join(str(row.get(field) or "") for field in text_fields)
        score = len(query_tokens & _significant_tokens(candidate_text))
        if score > best_score:
            best_score = score
            best_row = row
    return best_row


def _wikipedia_summary(query: str) -> Tuple[str | None, str | None]:
    search_url = (
        "https://en.wikipedia.org/w/api.php?action=opensearch&limit=3&namespace=0&format=json&search="
        f"{quote_plus(query)}"
    )
    payload = _fetch_json(search_url)
    if not payload or not isinstance(payload, list) or len(payload) < 2:
        return None, None

    titles = payload[1]
    if not isinstance(titles, list) or not titles:
        return None, None

    query_tokens = _significant_tokens(query, _QUERY_STOPWORDS)
    rows = [{"title": str(title or "").strip()} for title in titles if str(title or "").strip()]
    best = _best_matching_row(rows, query_tokens, ("title",))
    if best is None:
        return None, None

    page_title = str(best.get("title") or "").strip()
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
        "limit": "3",
    }
    url = f"https://www.wikidata.org/w/api.php?{urlencode(params)}"
    payload = _fetch_json(url)
    if not payload:
        return None, None

    rows = payload.get("search") or []
    if not isinstance(rows, list) or not rows:
        return None, None

    query_tokens = _significant_tokens(query, _QUERY_STOPWORDS)
    best = _best_matching_row(rows, query_tokens, ("label", "description"))
    if best is None:
        return None, None

    label = str(best.get("label") or "").strip()
    description = str(best.get("description") or "").strip()
    concept_url = str(best.get("concepturi") or "").strip() or None

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


def _strip_trailing_source(text: str, source_name: str) -> str:
    """Strip a redundant trailing publisher-name suffix from Google News text.

    Google News RSS titles are formatted as "Headline - Publisher Name", and
    descriptions carry the same publisher name in a trailing <font> tag that
    survives HTML-tag-stripping as a bare trailing word run (e.g. "...on June
    25  Rockstar Games"). The feed already exposes the publisher separately
    via <source>, so it's redundant clutter here - confirmed by fetching the
    raw feed directly during diagnosis.
    """
    text = text.strip()
    source_name = source_name.strip()
    if not text or not source_name:
        return text
    for suffix in (f" - {source_name}", f" {source_name}"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)].strip(" -:")
    return text


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
        source_name = str(item.findtext("source") or "").strip()
        description_raw = str(item.findtext("description") or "").strip()
        description = _strip_html_tags(html.unescape(description_raw)) if description_raw else None
        link = str(item.findtext("link") or "").strip() or None
        if not title:
            continue
        if source_name:
            title = _strip_trailing_source(title, source_name)
            if description:
                description = _strip_trailing_source(description, source_name)
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


# Comparison-shopping / SEO-ad CTA phrasing that shows up in unfiltered organic
# search scrapes (e.g. "OneClearWinner.com - Stop browsing online stores for
# hours - we are doing it for you") - real informative snippets don't talk
# like this regardless of topic.
_PROMOTIONAL_PATTERNS = [
    "stop browsing",
    "we are doing it for you",
    "lowest price",
    "best price",
    "top picks",
    "shop now",
    "buy now",
    "compare prices",
    "best deals",
    "free shipping",
    "% off",
    "save with",
    "limited time",
    "click here",
    "flexible monthly payment",
    "trade-in",
    "trade in and",
]

# Generic product-marketing filler: technically on-topic (mentions the brand
# name) but content-free boilerplate, not an actual answer to "what/who/when".
_MARKETING_FILLER_PATTERNS = [
    "helps you write, express yourself",
    "get things done effortlessly",
    "built for apple intelligence",
    "explore the latest",
    "helpful features like",
    "shop the latest",
]


def _looks_like_promotional(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return False
    return any(marker in normalized for marker in _PROMOTIONAL_PATTERNS) or any(
        marker in normalized for marker in _MARKETING_FILLER_PATTERNS
    )


def _dedupe_sources(sources: List[Dict[str, str | None]]) -> List[Dict[str, str | None]]:
    seen = set()
    deduped: List[Dict[str, str | None]] = []

    for source in sources:
        snippet = str(source.get("snippet") or "").strip()
        if not snippet:
            continue
        # Near-duplicate aware: the same noisy source often resurfaces with a
        # slightly different tail (e.g. "...for hours - we are doing it for
        # you." vs "...for hours."). Signature on the first ~8 significant
        # words instead of full text so both collapse to one entry.
        signature_tokens = re.sub(r"[^a-z0-9]+", " ", snippet.lower()).split()[:8]
        signature = " ".join(signature_tokens)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        deduped.append(source)

    return deduped


def _collect_sources(query: str) -> List[Dict[str, str | None]]:
    """Gather snippets from every provider.

    The providers are independent, so they run concurrently: serially this was
    up to six chained HTTP round trips (~30s worst case) on the turn thread.
    COLLECT_DEADLINE bounds the whole fan-out; stragglers are simply dropped.
    """

    def ddg_primary() -> List[Dict[str, str | None]]:
        text, url = _duckduckgo_summary(query)
        if not text:
            return []
        return [{"provider": "DuckDuckGo", "snippet": _clean_snippet(text), "url": url}]

    def ddg_related() -> List[Dict[str, str | None]]:
        return [
            {"provider": "DuckDuckGo related", "snippet": _clean_snippet(text), "url": url}
            for text, url in _duckduckgo_related_summaries(query, limit=3)
        ]

    def ddg_lite() -> List[Dict[str, str | None]]:
        return _ddg_lite_search(query, limit=3)

    def wikipedia() -> List[Dict[str, str | None]]:
        text, url = _wikipedia_summary(query)
        if not text:
            return []
        return [{"provider": "Wikipedia", "snippet": _clean_snippet(text), "url": url}]

    def wikidata() -> List[Dict[str, str | None]]:
        text, url = _wikidata_summary(query)
        if not text:
            return []
        return [{"provider": "Wikidata", "snippet": _clean_snippet(text), "url": url}]

    def google_news() -> List[Dict[str, str | None]]:
        if not _news_query(query):
            return []
        items = []
        for title, description, link in _google_news_highlights(query):
            # Prefer article descriptions over titles to avoid headline-only summaries.
            snippet_source = description if description and len(description.split()) >= 6 else title
            items.append(
                {
                    "provider": "Google News RSS",
                    "snippet": _clean_snippet(snippet_source, max_chars=260),
                    "url": link,
                }
            )
        return items

    # Order defines source precedence in the final answer.
    providers = [ddg_primary, ddg_lite, ddg_related, wikipedia, wikidata, google_news]
    results: List[List[Dict[str, str | None]]] = [[] for _ in providers]

    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {executor.submit(fn): index for index, fn in enumerate(providers)}
        deadline = time.monotonic() + COLLECT_DEADLINE
        for future in as_completed(futures, timeout=None):
            index = futures[future]
            provider_name = providers[index].__name__
            try:
                results[index] = future.result(timeout=max(0.0, deadline - time.monotonic()))
                _trace("provider_result", provider=provider_name, rows=len(results[index]))
            except Exception as exc:
                results[index] = []
                _trace_exc(exc, event="provider_failed", provider=provider_name)
            if time.monotonic() >= deadline:
                break

    sources = [item for group in results for item in group]
    sources = [item for item in sources if not _looks_like_promotional(str(item.get("snippet") or ""))]
    return _dedupe_sources(sources)


def _rank_sources_by_relevance(
    sources: List[Dict[str, str | None]], query: str
) -> List[Dict[str, str | None]]:
    """Rank sources by query relevance instead of fixed provider-scan order.

    Previously the first MAX_SOURCES_FOR_OVERVIEW sources in provider order
    (ddg_primary, ddg_lite, ...) were used regardless of how relevant they
    actually were - a low-value marketing snippet and a genuinely useful news
    snippet were weighted identically by position. Diversity pass caps each
    provider at 2 top slots so one noisy provider can't crowd out the rest.
    """
    query_tokens = _significant_tokens(query, _QUERY_STOPWORDS)

    def relevance(source: Dict[str, str | None]) -> int:
        if not query_tokens:
            return 0
        snippet = str(source.get("snippet") or "")
        return len(query_tokens & _significant_tokens(snippet))

    order = sorted(range(len(sources)), key=lambda i: (-relevance(sources[i]), i))

    ranked: List[Dict[str, str | None]] = []
    deferred: List[Dict[str, str | None]] = []
    provider_counts: Dict[str, int] = {}
    for i in order:
        provider = str(sources[i].get("provider") or "")
        if provider_counts.get(provider, 0) < 2:
            ranked.append(sources[i])
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        else:
            deferred.append(sources[i])

    return ranked + deferred


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


def _validate_synthesis(candidate: str) -> str | None:
    """Return None if candidate is a usable synthesis, else a rejection reason code.

    A short, direct, accurate answer ("The iPhone 17 is Apple's latest model,
    released in September 2025.") is strictly better than a longer extractive
    fragment dump, so this only rejects structural problems (echoed prompt
    scaffolding, raw URLs, empty/error output) - not brevity.
    """
    text = str(candidate or "").strip()
    if not text:
        return "empty"
    if _looks_like_llm_error(text):
        return "llm_error"

    word_count = len(text.split())
    if word_count < 8:
        return "too_short"
    if len(text) > 600:
        return "too_long"

    if not re.search(r"[.!?]", text):
        return "no_sentences"

    if "http://" in text.lower() or "https://" in text.lower():
        return "raw_url"

    lowered = text.lower()
    if "rules:" in lowered or "query:" in lowered:
        return "echoed_prompt"

    if re.search(r"(?m)^\[\d+\]\s", text):
        return "echoed_evidence"

    return None


def _synthesize_with_llm(
    query: str,
    sources: List[Dict[str, str | None]],
    llm: Any | None,
    compute_hint: str | None = None,
) -> Tuple[str | None, str]:
    """Returns (candidate_or_None, reason). reason is diagnostic even on success ("ok")."""
    if llm is None:
        return None, "llm_none"

    is_ready = getattr(llm, "is_ready", None)
    is_available = getattr(llm, "is_available", None)
    if callable(is_ready):
        try:
            ready = bool(is_ready())
        except Exception:
            return None, "llm_not_ready"

        if not ready:
            if callable(is_available):
                try:
                    if not bool(is_available()):
                        return None, "llm_not_ready"
                except Exception:
                    return None, "llm_not_ready"
            else:
                return None, "llm_not_ready"

    generate = getattr(llm, "generate", None)
    if not callable(generate):
        return None, "llm_not_ready"

    evidence_lines = []
    for index, source in enumerate(sources[:MAX_SOURCES_FOR_OVERVIEW], start=1):
        source_name = _source_display_name(source, index)
        snippet = str(source.get("snippet") or "").strip()
        if snippet:
            evidence_lines.append(f"[{index}] {source_name}: {snippet}")

    if not evidence_lines:
        return None, "no_evidence"

    prompt = (
        "Answer the query directly and concisely using only the provided sources.\n"
        "Rules:\n"
        "- Start with the direct answer in the first sentence.\n"
        "- Add up to 3 more sentences with specific supporting facts (names, dates, numbers, "
        "versions) drawn from the sources.\n"
        "- Sources are listed in order of relevance to the query - [1] is the most directly "
        "relevant. Build your answer around the most relevant, directly-on-topic sources first; "
        "don't let a less relevant source further down the list dominate the answer just because "
        "it sounds more novel or exciting.\n"
        "- Distinguish confirmed/currently-released facts from rumored, leaked, expected, or "
        "unreleased ones. Wording like \"rumored\", \"leaked\", \"expected to\", \"signaled\", or "
        "\"reportedly\" in a source means it is NOT confirmed. If the query asks what the current "
        "or latest state of something IS, answer with the confirmed/released reality first, and "
        "only mention rumored/upcoming items as a clearly-labeled aside if directly relevant - "
        "never present a rumor as settled fact.\n"
        "- If a source is only a headline-style teaser with a bare number claim (e.g. \"12 new "
        "features\", \"29 devices\") and no actual detail explaining it, don't repeat that number "
        "as if it were a concrete fact - skip it or mention it only as an unconfirmed headline.\n"
        "- Write like a neutral summary, not a description of the sources - never say things like "
        "\"according to the sources\" or list source names in the text.\n"
        "- Stay strictly faithful to the sources; mention uncertainty only if they genuinely conflict.\n"
        "- Include bracket citations such as [1] and [2] right after the specific facts they support.\n"
        "- Do not output raw URLs.\n\n"
        f"Query: {query}\n\n"
        "Sources:\n"
        + "\n".join(evidence_lines)
    )

    try:
        candidate = str(generate(prompt, None, device_hint=compute_hint, system_prompt=_SYNTHESIS_SYSTEM_PROMPT)).strip()
    except TypeError:
        try:
            candidate = str(generate(prompt, None)).strip()
        except Exception:
            return None, "generate_exception"
    except Exception:
        return None, "generate_exception"

    rejection = _validate_synthesis(candidate)
    if rejection:
        return None, f"validation_rejected:{rejection}"
    return candidate, "ok"


def _build_overview(
    query: str,
    sources: List[Dict[str, str | None]],
    llm: Any | None,
    compute_hint: str | None = None,
) -> Tuple[str, str, str]:
    """Synthesize with the LLM when possible, falling back to extractive splicing.

    Returns (overview, synthesis_mode, reason) - reason explains the mode:
    "ok" when LLM synthesis succeeded, or why it fell back to extractive
    (llm_none / llm_not_ready / no_evidence / generate_exception /
    validation_rejected:<check>) otherwise.
    """
    try:
        synthesized, reason = _synthesize_with_llm(query, sources, llm, compute_hint)
    except Exception as exc:
        _trace_exc(exc, event="synthesis_failed")
        synthesized, reason = None, "generate_exception"

    if synthesized:
        return synthesized, "llm", reason

    return _extractive_overview(sources, query), "extractive", reason


_NO_OVERVIEW_PLACEHOLDER = "I found web signals, but not enough high-quality detail to compose an overview yet."


def _extractive_sentences(
    sources: List[Dict[str, str | None]], query_tokens: set[str], require_relevance: bool
) -> List[str]:
    selected_sentences: List[str] = []
    seen: set[str] = set()

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
            if require_relevance and query_tokens and not (query_tokens & _significant_tokens(sentence)):
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

    return selected_sentences


def _extractive_overview(sources: List[Dict[str, str | None]], query: str = "") -> str:
    """Fallback for when LLM synthesis is unavailable: pick the most relevant
    sentences (sources are already relevance-ranked by the caller) rather than
    a fixed "first 2 sentences per source" scan, and prefer sentences that
    actually share a topic word with the query over generic on-brand filler.
    """
    query_tokens = _significant_tokens(query, _QUERY_STOPWORDS) if query else set()

    selected_sentences = _extractive_sentences(sources, query_tokens, require_relevance=True)
    if not selected_sentences:
        # Relevance filtering left nothing (e.g. very short query, or snippets
        # phrase things without literally repeating query words) - fall back
        # to the unfiltered pool rather than reporting failure.
        selected_sentences = _extractive_sentences(sources, query_tokens, require_relevance=False)

    if not selected_sentences:
        return _NO_OVERVIEW_PLACEHOLDER

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


def verified_answer(query: str, llm: Any | None = None, compute_hint: str | None = None) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return _response(False, "Please provide a query.")

    sources = _collect_sources(query)
    collected_count = len(sources)
    if not sources:
        _trace("sources_summary", collected=0, kept=0, synthesis="none", reason="no_sources")
        return _response(False, "I could not retrieve trusted web snippets right now.")

    query_tokens = _significant_tokens(query, _QUERY_STOPWORDS)
    if len(sources) < 2:
        single_snippet = str(sources[0].get("snippet") or "")
        if query_tokens and not (query_tokens & _significant_tokens(single_snippet)):
            sources = []

    snippets = [str(item.get("snippet") or "") for item in sources if str(item.get("snippet") or "").strip()]
    if not snippets:
        _trace("sources_summary", collected=collected_count, kept=0, synthesis="none", reason="quality_gate")
        return _response(False, "I could not retrieve trusted web snippets right now.")

    sources = _rank_sources_by_relevance(sources, query)
    confidence = _consistency_score(snippets)
    overview, synthesis_mode, synthesis_reason = _build_overview(query, sources, llm, compute_hint)

    if synthesis_mode == "extractive" and overview == _NO_OVERVIEW_PLACEHOLDER:
        _trace(
            "sources_summary",
            collected=collected_count,
            kept=len(sources),
            synthesis=synthesis_mode,
            synthesis_reason=synthesis_reason,
            reason="no_overview",
        )
        return _response(False, "I could not compose a reliable web overview right now.")

    # speech_text derives from overview alone - never included the label/
    # sources scaffolding below, so this split is unaffected by the display
    # formatting choices that follow.
    speech_text = re.sub(r"\s+", " ", re.sub(r"\s*\[\d+\]", "", overview)).strip()

    # Minimal chat formatting: no redundant "AI web overview:" label (the chat
    # bubble already shows this is JARVIS talking), a blank line before the
    # meta footer, sources on one tidy line, and a caveat line only when
    # confidence is actually low - high/moderate agreement is the normal case
    # and doesn't need flagging every response.
    footer_lines: List[str] = []

    if confidence not in ("high", "moderate"):
        footer_lines.append("Limited sources - worth double-checking.")

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
        footer_lines.append(f"Sources: {' · '.join(source_names)}")

    source_urls = [str(item.get("url") or "").strip() for item in sources if str(item.get("url") or "").strip()]

    _trace(
        "sources_summary",
        collected=collected_count,
        kept=len(sources),
        synthesis=synthesis_mode,
        synthesis_reason=synthesis_reason,
        consistency=confidence,
    )

    display_text = overview
    if footer_lines:
        display_text = overview + "\n\n" + "\n".join(footer_lines)

    return _response(
        True,
        display_text,
        {
            "mode": "verified_web_overview",
            "synthesis": synthesis_mode,
            "consistency": confidence,
            "sources": source_names,
            "source_urls": source_urls,
            "snippets": snippets,
            "providers": [str(item.get("provider") or "") for item in sources],
            "speech_text": speech_text,
        },
    )
