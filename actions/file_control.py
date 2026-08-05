from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from actions.common import response as _response
from platform_adapters import backend


SEARCH_DIRS = [Path.home(), Path.home() / "Desktop", Path.home() / "Documents", Path.home() / "Downloads"]

# A fuzzy name match from speech must never launch something executable;
# only an exact path typed/spoken in full may open these.
BLOCKED_EXECUTABLE_EXTS = {".exe", ".bat", ".cmd", ".com", ".msi", ".ps1", ".sh", ".desktop", ".appimage", ".run"}


def _find_file(name_or_path: str) -> Path | None:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    needle = name_or_path.lower()
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        try:
            for file in root.rglob("*"):
                if needle in file.name.lower():
                    return file
        except Exception:
            continue
    return None


def handle(entities: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = entities.get("file_action", "open")
        query = entities.get("file_path") or entities.get("file_name") or entities.get("search_query")

        if not query:
            return _response(False, "Please specify a file or folder name.")

        path = _find_file(str(query))
        if path is None:
            return _response(False, f"I could not find {query}.")

        if action in {"open", "launch"}:
            exact_path_given = Path(str(query)).exists()
            if not exact_path_given and path.suffix.lower() in BLOCKED_EXECUTABLE_EXTS:
                return _response(
                    False,
                    f"{path.name} looks executable, so I won't open it from a name match. "
                    "Give me its full path if you really want to run it.",
                    {"path": str(path), "blocked": True},
                )
            backend.open_path(str(path))
            return _response(True, f"Opened {path.name}.", {"path": str(path)})

        if action in {"read", "show"} and path.is_file():
            content = path.read_text(encoding="utf-8", errors="ignore")
            preview = content[:1500]
            return _response(True, f"Here is the content of {path.name}.", {"path": str(path), "content": preview})

        if action == "find":
            return _response(True, f"Found {path.name}.", {"path": str(path)})

        return _response(False, f"Unsupported file action: {action}.")
    except Exception as exc:
        return _response(False, f"I could not complete the file action: {exc}", {"error": str(exc)})
