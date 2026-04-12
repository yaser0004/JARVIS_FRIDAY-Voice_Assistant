from __future__ import annotations

import re
import webbrowser
from typing import Any, Dict
from urllib.parse import quote_plus


PLATFORM_URLS = {
    "google": "https://google.com/search?q={}",
    "youtube": "https://youtube.com/results?search_query={}",
    "reddit": "https://reddit.com/search/?q={}",
    "twitter": "https://twitter.com/search?q={}",
    "amazon": "https://amazon.in/s?k={}",
    "flipkart": "https://flipkart.com/search?q={}",
    "github": "https://github.com/search?q={}",
    "stackoverflow": "https://stackoverflow.com/search?q={}",
    "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search={}",
    "maps": "https://maps.google.com/search?q={}",
}

SITE_MAP = {
    "google": "https://google.com",
    "youtube": "https://youtube.com",
    "reddit": "https://reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://twitter.com",
    "amazon": "https://amazon.in",
    "flipkart": "https://flipkart.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "wikipedia": "https://wikipedia.org",
    "gmail": "https://mail.google.com",
    "drive": "https://drive.google.com",
    "docs": "https://docs.google.com",
    "sheets": "https://sheets.google.com",
    "slides": "https://slides.google.com",
    "calendar": "https://calendar.google.com",
    "maps": "https://maps.google.com",
    "linkedin": "https://linkedin.com",
    "instagram": "https://instagram.com",
    "facebook": "https://facebook.com",
    "netflix": "https://netflix.com",
    "prime": "https://primevideo.com",
    "hotstar": "https://hotstar.com",
    "disney": "https://disneyplus.com",
    "spotify": "https://open.spotify.com",
    "soundcloud": "https://soundcloud.com",
    "apple music": "https://music.apple.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "perplexity": "https://perplexity.ai",
    "huggingface": "https://huggingface.co",
    "kaggle": "https://kaggle.com",
    "colab": "https://colab.research.google.com",
    "leetcode": "https://leetcode.com",
    "codeforces": "https://codeforces.com",
    "hackerrank": "https://hackerrank.com",
    "geeksforgeeks": "https://geeksforgeeks.org",
    "medium": "https://medium.com",
    "devto": "https://dev.to",
    "notion": "https://notion.so",
    "figma": "https://figma.com",
    "canva": "https://canva.com",
    "dribbble": "https://dribbble.com",
    "behance": "https://behance.net",
    "pinterest": "https://pinterest.com",
    "quora": "https://quora.com",
    "whatsapp": "https://web.whatsapp.com",
    "telegram": "https://web.telegram.org",
    "discord": "https://discord.com/app",
    "slack": "https://slack.com/signin",
    "teams": "https://teams.microsoft.com",
    "zoom": "https://zoom.us",
    "trello": "https://trello.com",
    "asana": "https://asana.com",
    "jira": "https://atlassian.com/software/jira",
    "confluence": "https://atlassian.com/software/confluence",
    "bitbucket": "https://bitbucket.org",
    "gitlab": "https://gitlab.com",
    "npm": "https://npmjs.com",
    "pypi": "https://pypi.org",
    "docker hub": "https://hub.docker.com",
    "cloudflare": "https://cloudflare.com",
    "aws": "https://aws.amazon.com",
    "azure": "https://portal.azure.com",
    "gcp": "https://console.cloud.google.com",
    "digitalocean": "https://digitalocean.com",
    "linode": "https://linode.com",
    "vercel": "https://vercel.com",
    "netlify": "https://netlify.com",
    "railway": "https://railway.app",
    "render": "https://render.com",
    "namecheap": "https://namecheap.com",
    "godaddy": "https://godaddy.com",
    "udemy": "https://udemy.com",
    "coursera": "https://coursera.org",
    "edx": "https://edx.org",
    "mit ocw": "https://ocw.mit.edu",
    "arxiv": "https://arxiv.org",
    "researchgate": "https://researchgate.net",
    "ieee": "https://ieeexplore.ieee.org",
    "springer": "https://springer.com",
    "sciencedirect": "https://sciencedirect.com",
    "nature": "https://nature.com",
    "bbc": "https://bbc.com",
    "cnn": "https://cnn.com",
    "ndtv": "https://ndtv.com",
    "times of india": "https://timesofindia.indiatimes.com",
    "the hindu": "https://thehindu.com",
    "moneycontrol": "https://moneycontrol.com",
    "tradingview": "https://tradingview.com",
    "coinmarketcap": "https://coinmarketcap.com",
    "binance": "https://binance.com",
    "bookmyshow": "https://bookmyshow.com",
    "zomato": "https://zomato.com",
    "swiggy": "https://swiggy.com",
    "ubereats": "https://ubereats.com",
    "uber": "https://uber.com",
    "ola": "https://olacabs.com",
    "booking": "https://booking.com",
    "airbnb": "https://airbnb.com",
    "tripadvisor": "https://tripadvisor.com",
    "skyscanner": "https://skyscanner.com",
    "irctc": "https://irctc.co.in",
    "paytm": "https://paytm.com",
    "phonepe": "https://phonepe.com",
    "gpay": "https://pay.google.com",
    "paypal": "https://paypal.com",
    "dropbox": "https://dropbox.com",
    "onedrive": "https://onedrive.live.com",
    "mega": "https://mega.nz",
}

URL_LIKE = re.compile(r"^(https?://|www\.|\w+\.(com|org|net|io|edu))(.*)$", re.IGNORECASE)


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def search(query: str, platform: str = "google") -> Dict[str, Any]:
    try:
        clean_query = (query or "").strip()
        if not clean_query:
            return _response(False, "Please provide a search query.")

        key = (platform or "google").strip().lower()
        template = PLATFORM_URLS.get(key, PLATFORM_URLS["google"])
        url = template.format(quote_plus(clean_query))
        webbrowser.open(url)
        return _response(True, f"Searching {key} for {clean_query}.", {"url": url, "platform": key})
    except Exception as exc:
        return _response(False, f"I could not run the search: {exc}", {"error": str(exc)})


def open_url(url_or_name: str) -> Dict[str, Any]:
    try:
        token = (url_or_name or "").strip().lower()
        if not token:
            return _response(False, "Please provide a website name or URL.")

        if URL_LIKE.match(token):
            url = token if token.startswith("http") else f"https://{token}"
        else:
            url = SITE_MAP.get(token)
            if not url:
                normalized = token.replace(" ", "")
                for key, value in SITE_MAP.items():
                    if normalized == key.replace(" ", ""):
                        url = value
                        break
            if not url:
                return _response(False, f"I do not have a website mapping for {token}.")

        webbrowser.open(url)
        return _response(True, f"Opening {token}.", {"url": url})
    except Exception as exc:
        return _response(False, f"I could not open that website: {exc}", {"error": str(exc)})
