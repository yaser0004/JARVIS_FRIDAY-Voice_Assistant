from __future__ import annotations

import json
import os
import re
import string
import subprocess
import time
import webbrowser
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pygame

from actions import app_control
from core.config import APPDATA_DIR


AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aac", ".wma", ".opus"}
MEDIA_INDEX_PATH = APPDATA_DIR / "media_index.json"
MEDIA_INDEX_VERSION = 1
REQUEST_TIMEOUT_SECONDS = 10

EXCLUDED_DIR_NAMES = {
    "$recycle.bin",
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "appdata",
    "node_modules",
    ".git",
    "temp",
}

PLATFORM_ALIASES = {
    "auto": "auto",
    "local": "local",
    "disk": "local",
    "pc": "local",
    "drive": "local",
    "spotify": "spotify",
    "apple music": "apple_music",
    "apple": "apple_music",
    "itunes": "apple_music",
    "youtube": "youtube",
    "youtube music": "youtube_music",
    "ytmusic": "youtube_music",
    "soundcloud": "soundcloud",
    "deezer": "deezer",
    "tidal": "tidal",
    "amazon music": "amazon_music",
    "prime music": "amazon_music",
    "gaana": "gaana",
    "jiosaavn": "jiosaavn",
}


class LocalMediaLibrary:
    def __init__(self) -> None:
        self.files: List[Path] = []
        self.current_index: int = -1
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        self.refresh()

    def refresh(self, force_rescan: bool = False) -> None:
        index = ensure_media_index(force_rescan=force_rescan)
        self.files = [Path(item["path"]) for item in index.get("tracks", []) if Path(item.get("path", "")).exists()]

    def find(self, title: str) -> Optional[Path]:
        query = _normalize_text(title)
        if not query:
            return None

        best: tuple[float, Path] | None = None
        for file in self.files:
            stem = _normalize_text(file.stem)
            if not stem:
                continue

            if query == stem:
                return file

            if query in stem:
                score = 0.95
            else:
                score = SequenceMatcher(a=query, b=stem).ratio()

            if best is None or score > best[0]:
                best = (score, file)

        if best is not None and best[0] >= 0.70:
            return best[1]
        return None

    def play_file(self, file: Path) -> None:
        pygame.mixer.music.load(str(file))
        pygame.mixer.music.play()
        if file in self.files:
            self.current_index = self.files.index(file)

    def pause(self) -> None:
        pygame.mixer.music.pause()

    def resume(self) -> None:
        pygame.mixer.music.unpause()

    def stop(self) -> None:
        pygame.mixer.music.stop()

    def next(self) -> Optional[Path]:
        if not self.files:
            return None
        self.current_index = (self.current_index + 1) % len(self.files)
        track = self.files[self.current_index]
        self.play_file(track)
        return track

    def previous(self) -> Optional[Path]:
        if not self.files:
            return None
        self.current_index = (self.current_index - 1) % len(self.files)
        track = self.files[self.current_index]
        self.play_file(track)
        return track


_LIBRARY: LocalMediaLibrary | None = None


def _library() -> LocalMediaLibrary:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = LocalMediaLibrary()
    return _LIBRARY


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _normalize_platform(platform: str) -> str:
    key = _normalize_text(platform)
    return PLATFORM_ALIASES.get(key, key or "auto")


def _list_drive_roots() -> List[Path]:
    if os.name != "nt":
        return [Path.home()]

    roots: List[Path] = []
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:\\")
        if root.exists():
            roots.append(root)

    if not roots:
        roots = [Path.home()]
    return roots


def _iter_audio_files(root: Path):
    try:
        for current_root, dirs, files in os.walk(root, topdown=True):
            dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIR_NAMES]
            for file_name in files:
                suffix = Path(file_name).suffix.lower()
                if suffix in AUDIO_EXTENSIONS:
                    yield Path(current_root) / file_name
    except Exception:
        return


def build_media_index() -> Dict[str, Any]:
    tracks: List[Dict[str, str]] = []
    seen: set[str] = set()

    for root in _list_drive_roots():
        for audio_path in _iter_audio_files(root):
            resolved = str(audio_path)
            key = resolved.lower()
            if key in seen:
                continue
            seen.add(key)
            tracks.append(
                {
                    "path": resolved,
                    "title": audio_path.stem,
                    "title_norm": _normalize_text(audio_path.stem),
                }
            )

    return {
        "version": MEDIA_INDEX_VERSION,
        "generated_at": time.time(),
        "stats": {
            "tracks": len(tracks),
            "drives_scanned": len(_list_drive_roots()),
        },
        "tracks": tracks,
    }


def _load_media_index() -> Dict[str, Any] | None:
    if not MEDIA_INDEX_PATH.exists():
        return None
    try:
        with MEDIA_INDEX_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data.get("version", 0)) != MEDIA_INDEX_VERSION:
            return None
        if not isinstance(data.get("tracks"), list):
            return None
        return data
    except Exception:
        return None


def _save_media_index(index: Dict[str, Any]) -> None:
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    with MEDIA_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def ensure_media_index(force_rescan: bool = False) -> Dict[str, Any]:
    if not force_rescan:
        existing = _load_media_index()
        if existing is not None:
            return existing

    index = build_media_index()
    _save_media_index(index)
    return index


def rescan_media_index() -> Dict[str, Any]:
    index = ensure_media_index(force_rescan=True)
    stats = index.get("stats", {})
    if _LIBRARY is not None:
        _LIBRARY.refresh(force_rescan=False)
    return _response(
        True,
        f"Music index refreshed. Indexed {stats.get('tracks', 0)} tracks.",
        {"index_path": str(MEDIA_INDEX_PATH), **stats},
    )


def media_index_summary() -> Dict[str, Any]:
    index = ensure_media_index()
    stats = index.get("stats", {})
    return _response(
        True,
        f"Indexed {stats.get('tracks', 0)} tracks.",
        {"index_path": str(MEDIA_INDEX_PATH), **stats},
    )


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def _open_spotify_search(title: str) -> None:
    uri = f"spotify:search:{quote_plus(title)}"
    subprocess.Popen(["cmd", "/c", "start", "", uri], shell=False)


def _youtube_first_video_url(query: str) -> str | None:
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    request = Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        html = response.read().decode("utf-8", errors="ignore")

    match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
    if not match:
        return None
    video_id = match.group(1)
    return f"https://www.youtube.com/watch?v={video_id}&autoplay=1"


def _open_web_platform(platform: str, query: str) -> bool:
    encoded = quote_plus(query)
    urls = {
        "spotify": f"https://open.spotify.com/search/{encoded}",
        "apple_music": f"https://music.apple.com/us/search?term={encoded}",
        "youtube": f"https://www.youtube.com/results?search_query={encoded}",
        "youtube_music": f"https://music.youtube.com/search?q={encoded}",
        "soundcloud": f"https://soundcloud.com/search?q={encoded}",
        "deezer": f"https://www.deezer.com/search/{encoded}",
        "tidal": f"https://listen.tidal.com/search/tracks?q={encoded}",
        "amazon_music": f"https://music.amazon.com/search/{encoded}",
        "gaana": f"https://gaana.com/search/{encoded}",
        "jiosaavn": f"https://www.jiosaavn.com/search/{encoded}",
    }
    url = urls.get(platform)
    if not url:
        return False
    return bool(webbrowser.open(url))


def _play_from_platform(query: str, platform: str) -> Dict[str, Any]:
    normalized = _normalize_platform(platform)

    if normalized in {"youtube", "youtube_music"}:
        try:
            direct = _youtube_first_video_url(query)
            if direct:
                webbrowser.open(direct)
                return _response(True, f"Playing {query} on YouTube.", {"platform": "youtube", "url": direct})
        except Exception:
            pass
        webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(query)}")
        return _response(True, f"Searching YouTube for {query}.", {"platform": "youtube"})

    if normalized == "spotify":
        if app_control.is_app_available("spotify"):
            _open_spotify_search(query)
            return _response(True, f"Playing {query} on Spotify.", {"platform": "spotify", "mode": "native"})
        if _open_web_platform("spotify", query):
            return _response(True, f"Playing {query} on Spotify web.", {"platform": "spotify", "mode": "web"})

    if normalized == "apple_music":
        if app_control.is_app_available("apple music") or app_control.is_app_available("itunes"):
            subprocess.Popen(["cmd", "/c", "start", "", f"https://music.apple.com/us/search?term={quote_plus(query)}"], shell=False)
            return _response(True, f"Playing {query} on Apple Music.", {"platform": "apple_music", "mode": "native"})
        if _open_web_platform("apple_music", query):
            return _response(True, f"Playing {query} on Apple Music web.", {"platform": "apple_music", "mode": "web"})

    if normalized in {"soundcloud", "deezer", "tidal", "amazon_music", "gaana", "jiosaavn"}:
        if _open_web_platform(normalized, query):
            return _response(True, f"Playing {query} on {normalized.replace('_', ' ')}.", {"platform": normalized})

    # Final fallback: direct YouTube autoplay attempt.
    try:
        direct = _youtube_first_video_url(query)
        if direct:
            webbrowser.open(direct)
            return _response(True, f"Could not use {platform}; playing {query} on YouTube.", {"platform": "youtube", "url": direct})
    except Exception:
        pass

    webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(query)}")
    return _response(True, f"Could not use {platform}; searching YouTube for {query}.", {"platform": "youtube"})


def play(title: str, platform: str = "auto") -> Dict[str, Any]:
    try:
        query = (title or "").strip()
        if not query:
            return _response(False, "Please tell me what to play.")

        library = _library()

        target = _normalize_platform(platform)

        if target in {"local", "auto"}:
            match = library.find(query)
            if match:
                library.play_file(match)
                return _response(True, f"Playing {match.stem} from disk.", {"path": str(match), "platform": "local"})
            return _play_from_platform(query, "youtube")

        return _play_from_platform(query, target)
    except Exception as exc:
        return _response(False, f"I could not play media: {exc}", {"error": str(exc)})


def pause() -> Dict[str, Any]:
    try:
        _library().pause()
        return _response(True, "Paused playback.")
    except Exception as exc:
        return _response(False, f"I could not pause playback: {exc}", {"error": str(exc)})


def resume() -> Dict[str, Any]:
    try:
        _library().resume()
        return _response(True, "Resumed playback.")
    except Exception as exc:
        return _response(False, f"I could not resume playback: {exc}", {"error": str(exc)})


def stop() -> Dict[str, Any]:
    try:
        _library().stop()
        return _response(True, "Stopped playback.")
    except Exception as exc:
        return _response(False, f"I could not stop playback: {exc}", {"error": str(exc)})


def next_track() -> Dict[str, Any]:
    try:
        track = _library().next()
        if track is None:
            return _response(False, "No local tracks are indexed yet.")
        return _response(True, f"Playing next track: {track.stem}", {"path": str(track)})
    except Exception as exc:
        return _response(False, f"I could not move to the next track: {exc}", {"error": str(exc)})


def previous_track() -> Dict[str, Any]:
    try:
        track = _library().previous()
        if track is None:
            return _response(False, "No local tracks are indexed yet.")
        return _response(True, f"Playing previous track: {track.stem}", {"path": str(track)})
    except Exception as exc:
        return _response(False, f"I could not move to the previous track: {exc}", {"error": str(exc)})
