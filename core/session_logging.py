from __future__ import annotations

import atexit
import io
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from core.config import ROOT_DIR


class _TeeTextStream(io.TextIOBase):
    def __init__(self, primary, mirror) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, s: str) -> int:
        text = str(s)
        self._primary.write(text)
        self._mirror.write(text)
        return len(text)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        method = getattr(self._primary, "isatty", None)
        return bool(method()) if callable(method) else False

    @property
    def encoding(self) -> str:
        return str(getattr(self._primary, "encoding", "utf-8") or "utf-8")


def _sanitize_for_json(value: Any, *, max_string: int = 4000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, str):
        text = value
        if len(text) > max_string:
            return text[:max_string] + f"...<truncated:{len(text) - max_string}>"
        return text

    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"

    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item, max_string=max_string) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item, max_string=max_string) for item in value]

    return _sanitize_for_json(str(value), max_string=max_string)


_ACTIVE_LOGGER_LOCK = threading.Lock()
_ACTIVE_LOGGER: "SessionTerminalLogger | None" = None


def set_active_session_logger(logger: "SessionTerminalLogger | None") -> None:
    global _ACTIVE_LOGGER
    with _ACTIVE_LOGGER_LOCK:
        _ACTIVE_LOGGER = logger


def get_active_session_logger() -> "SessionTerminalLogger | None":
    with _ACTIVE_LOGGER_LOCK:
        return _ACTIVE_LOGGER


def trace_event(component: str, kind: str, **details: Any) -> None:
    logger = get_active_session_logger()
    if logger is None:
        return
    logger.trace_event(component, kind, **details)


def trace_exception(component: str, exc: BaseException, **details: Any) -> None:
    logger = get_active_session_logger()
    if logger is None:
        return
    logger.trace_exception(component, exc, **details)


class SessionTerminalLogger:
    def __init__(self, logs_dir: Path | None = None) -> None:
        target_dir = logs_dir or (ROOT_DIR / "logs")
        target_dir.mkdir(parents=True, exist_ok=True)

        self._target_dir = target_dir
        self._start_dt = datetime.now()
        self._start_stamp = self._start_dt.strftime("%Y-%m-%d %H-%M-%S")
        self._active_path = target_dir / f"{self._start_stamp} -- active.log"
        self._active_trace_path = target_dir / f"{self._start_stamp} -- active.trace.jsonl"
        self._stream = self._active_path.open("w", encoding="utf-8", buffering=1)
        self._trace_stream = self._active_trace_path.open("w", encoding="utf-8", buffering=1)
        self._trace_lock = threading.Lock()
        self._event_seq = 0
        self._pid = int(getattr(os, "getpid", lambda: -1)())

        self._stdout_original = sys.stdout
        self._stderr_original = sys.stderr
        self._sys_excepthook_original = getattr(sys, "excepthook", None)
        self._threading_excepthook_original = getattr(threading, "excepthook", None)

        sys.stdout = _TeeTextStream(self._stdout_original, self._stream)
        sys.stderr = _TeeTextStream(self._stderr_original, self._stream)

        self._install_exception_hooks()

        self._stream.write(f"[SESSION START] {self._start_dt.isoformat()}\n")
        self._closed = False
        set_active_session_logger(self)
        self.trace_event(
            "session",
            "start",
            start_iso=self._start_dt.isoformat(),
            terminal_log=str(self._active_path),
            trace_log=str(self._active_trace_path),
            argv=list(sys.argv),
        )
        atexit.register(self._atexit_stop)

    def _install_exception_hooks(self) -> None:
        original_sys_hook = self._sys_excepthook_original

        def _sys_hook(exc_type, exc_value, exc_tb) -> None:
            try:
                if isinstance(exc_value, BaseException):
                    self.trace_exception(
                        "python.sys_excepthook",
                        exc_value,
                        traceback="".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
                    )
            except Exception:
                pass

            if callable(original_sys_hook):
                original_sys_hook(exc_type, exc_value, exc_tb)

        sys.excepthook = _sys_hook

        original_thread_hook = self._threading_excepthook_original
        if callable(original_thread_hook):

            def _thread_hook(args) -> None:
                try:
                    self.trace_exception(
                        "python.thread_excepthook",
                        args.exc_value,
                        thread_name=str(getattr(args.thread, "name", "")),
                        traceback="".join(
                            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
                        ),
                    )
                except Exception:
                    pass
                original_thread_hook(args)

            threading.excepthook = _thread_hook

    def _restore_exception_hooks(self) -> None:
        if callable(self._sys_excepthook_original):
            sys.excepthook = self._sys_excepthook_original
        if callable(self._threading_excepthook_original):
            threading.excepthook = self._threading_excepthook_original

    def _next_sequence(self) -> int:
        with self._trace_lock:
            self._event_seq += 1
            return self._event_seq

    def trace_event(self, component: str, kind: str, **details: Any) -> None:
        if self._closed:
            return

        payload: Dict[str, Any] = {
            "seq": self._next_sequence(),
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "epoch_ms": int(time.time() * 1000),
            "process_id": self._pid,
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "source": str(component or "unknown"),
            "event": str(kind or "unknown"),
            "details": _sanitize_for_json(details),
        }

        line = json.dumps(payload, ensure_ascii=True)
        with self._trace_lock:
            self._trace_stream.write(line + "\n")

    def trace_exception(self, component: str, exc: BaseException, **details: Any) -> None:
        stack = ""
        explicit_trace = details.pop("traceback", "")
        context_event = str(details.pop("event", "exception") or "exception")
        context_source = details.pop("source", None)
        if context_source is not None:
            details["context_source"] = context_source
        if explicit_trace:
            stack = str(explicit_trace)
        else:
            stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        self.trace_event(
            component,
            context_event,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=stack,
            **details,
        )

    @property
    def trace_path(self) -> Path:
        return self._active_trace_path

    @property
    def terminal_path(self) -> Path:
        return self._active_path

    def _resolve_final_paths(self, end_stamp: str) -> tuple[Path, Path]:
        final_terminal_path = self._target_dir / f"{self._start_stamp} -- {end_stamp}.log"
        final_trace_path = self._target_dir / f"{self._start_stamp} -- {end_stamp}.trace.jsonl"

        if final_terminal_path.exists() or final_trace_path.exists():
            suffix = 1
            while True:
                candidate_terminal = self._target_dir / f"{self._start_stamp} -- {end_stamp} ({suffix}).log"
                candidate_trace = self._target_dir / f"{self._start_stamp} -- {end_stamp} ({suffix}).trace.jsonl"
                if not candidate_terminal.exists() and not candidate_trace.exists():
                    final_terminal_path = candidate_terminal
                    final_trace_path = candidate_trace
                    break
                suffix += 1

        return final_terminal_path, final_trace_path

    def _atexit_stop(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    def stop(self) -> Path:
        if self._closed:
            return self._active_path

        end_dt = datetime.now()
        end_stamp = end_dt.strftime("%Y-%m-%d %H-%M-%S")
        self.trace_event("session", "stop_requested", end_iso=end_dt.isoformat())

        self._stream.write(f"[SESSION END] {end_dt.isoformat()}\n")
        sys.stdout = self._stdout_original
        sys.stderr = self._stderr_original
        self._restore_exception_hooks()
        self._stream.flush()
        self._trace_stream.flush()
        self._stream.close()
        self._trace_stream.close()

        final_path, final_trace_path = self._resolve_final_paths(end_stamp)
        self._active_path.rename(final_path)
        self._active_trace_path.rename(final_trace_path)

        self._active_path = final_path
        self._active_trace_path = final_trace_path
        self._closed = True

        logger = get_active_session_logger()
        if logger is self:
            set_active_session_logger(None)

        return final_path
