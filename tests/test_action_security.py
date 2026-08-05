"""Security regressions for action handlers driven by speech transcripts."""

import pytest

from actions import file_control, media_control


def test_spotify_search_does_not_shell_out(monkeypatch):
    """A media title must never reach cmd.exe, where & | ^ are metacharacters."""
    opened = []
    monkeypatch.setattr(media_control.webbrowser, "open", lambda uri: opened.append(uri) or True)

    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("media_control must not spawn a subprocess for URI opening")

    monkeypatch.setattr("subprocess.Popen", explode)

    media_control._open_spotify_search("song & rm -rf | evil ^ payload")

    assert len(opened) == 1
    assert opened[0].startswith("spotify:search:")
    # quote_plus percent-encodes the metacharacters instead of passing them on.
    for char in ("&", "|", "^"):
        assert char not in opened[0]


@pytest.mark.parametrize("suffix", [".sh", ".exe", ".desktop", ".bat", ".ps1"])
def test_fuzzy_name_match_refuses_executables(monkeypatch, tmp_path, suffix):
    target = tmp_path / f"payload{suffix}"
    target.write_text("#!/bin/sh\necho pwned\n")

    monkeypatch.setattr(file_control, "_find_file", lambda query: target)
    opened = []
    monkeypatch.setattr(file_control.backend, "open_path", lambda p: opened.append(p), raising=False)

    result = file_control.handle({"file_action": "open", "file_name": "payload"})

    assert result["success"] is False
    assert result["data"]["blocked"] is True
    assert opened == []


def test_exact_path_to_executable_is_allowed(monkeypatch, tmp_path):
    target = tmp_path / "tool.sh"
    target.write_text("#!/bin/sh\necho hi\n")

    monkeypatch.setattr(file_control, "_find_file", lambda query: target)
    opened = []
    monkeypatch.setattr(file_control.backend, "open_path", lambda p: opened.append(p), raising=False)

    result = file_control.handle({"file_action": "open", "file_path": str(target)})

    assert result["success"] is True
    assert opened == [str(target)]


def test_ordinary_document_still_opens(monkeypatch, tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("hello")

    monkeypatch.setattr(file_control, "_find_file", lambda query: target)
    opened = []
    monkeypatch.setattr(file_control.backend, "open_path", lambda p: opened.append(p), raising=False)

    result = file_control.handle({"file_action": "open", "file_name": "notes"})

    assert result["success"] is True
    assert opened == [str(target)]
