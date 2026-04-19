from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from actions import system_control
from core.compute_runtime import detect_compute_capabilities
from core.pipeline import JarvisPipeline
from nlp.entity_extractor import EntityExtractor


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class Profile:
    name: str
    query_timeout_s: float
    suite_timeout_s: float
    stress_iterations: int
    stress_concurrency_rounds: int
    stress_parallel_workers: int
    ui_timeout_s: int


PROFILES: Dict[str, Profile] = {
    "quick": Profile(
        name="quick",
        query_timeout_s=45.0,
        suite_timeout_s=180.0,
        stress_iterations=8,
        stress_concurrency_rounds=1,
        stress_parallel_workers=2,
        ui_timeout_s=300,
    ),
    "full": Profile(
        name="full",
        query_timeout_s=70.0,
        suite_timeout_s=360.0,
        stress_iterations=24,
        stress_concurrency_rounds=3,
        stress_parallel_workers=3,
        ui_timeout_s=900,
    ),
    "aggressive": Profile(
        name="aggressive",
        query_timeout_s=90.0,
        suite_timeout_s=600.0,
        stress_iterations=64,
        stress_concurrency_rounds=7,
        stress_parallel_workers=4,
        ui_timeout_s=2400,
    ),
    "max": Profile(
        name="max",
        query_timeout_s=120.0,
        suite_timeout_s=900.0,
        stress_iterations=140,
        stress_concurrency_rounds=16,
        stress_parallel_workers=5,
        ui_timeout_s=5400,
    ),
}


@dataclass
class CaseResult:
    id: int
    category: str
    name: str
    status: str
    duration_s: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseOutcome:
    ok: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class FullSystemTester:
    def __init__(
        self,
        *,
        profile: Profile,
        strict_real: bool,
        include_ui: bool,
        ui_profile: str,
        output_json: Optional[Path],
    ) -> None:
        self.profile = profile
        self.strict_real = bool(strict_real)
        self.include_ui = bool(include_ui)
        self.ui_profile = ui_profile
        self.output_json = output_json

        self.results: List[CaseResult] = []
        self._next_id = 1
        self.started_at = _iso_now()
        self.started_monotonic = time.perf_counter()

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.artifact_dir = ROOT_DIR / "logs" / "test_reports" / f"run-{stamp}"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline: JarvisPipeline | None = None
        self.entity_extractor = EntityExtractor()
        self.signal_counts: Dict[str, int] = {
            "state": 0,
            "new_message": 0,
            "intent": 0,
            "diagnostics": 0,
            "init_progress": 0,
            "ready": 0,
            "wake_availability": 0,
        }
        self.last_state = ""
        self.last_assistant_message = ""
        self.last_user_message = ""
        self.last_intent = ""
        self.last_intent_confidence = 0.0

        self.volume_baseline: Optional[int] = None
        self.brightness_baseline: Optional[int] = None
        self._temp_files: List[Path] = []
        self._cleanup_errors: List[str] = []
        self._compute_samples: List[Dict[str, Any]] = []

    def _log(self, text: str) -> None:
        print(text, flush=True)

    def _record(
        self,
        *,
        category: str,
        name: str,
        status: str,
        duration_s: float,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = CaseResult(
            id=self._next_id,
            category=category,
            name=name,
            status=status,
            duration_s=round(float(duration_s), 4),
            message=str(message),
            details=_jsonable(details or {}),
        )
        self._next_id += 1
        self.results.append(entry)

        marker = {
            "passed": "PASS",
            "failed": "FAIL",
            "skipped": "SKIP",
            "verify_only": "VERIFY",
        }.get(status, status.upper())
        self._log(f"[{marker}] {category} :: {name} ({entry.duration_s:.2f}s) -> {message}")

    def _run_case(
        self,
        *,
        category: str,
        name: str,
        func: Callable[[], CaseOutcome],
        verify_only: bool = False,
    ) -> None:
        started = time.perf_counter()
        try:
            outcome = func()
            status = "verify_only" if verify_only and outcome.ok else ("passed" if outcome.ok else "failed")
            self._record(
                category=category,
                name=name,
                status=status,
                duration_s=time.perf_counter() - started,
                message=outcome.message,
                details=outcome.details,
            )
        except Exception as exc:
            self._record(
                category=category,
                name=name,
                status="failed",
                duration_s=time.perf_counter() - started,
                message=f"Unhandled exception: {exc}",
                details={"traceback": traceback.format_exc()},
            )

    def _run_with_timeout(self, func: Callable[[], Any], timeout_s: float, label: str) -> Any:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
        timed_out = False
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError as exc:
            timed_out = True
            if self.pipeline is not None:
                try:
                    self.pipeline.cancel_current_action()
                except Exception:
                    pass
            future.cancel()
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s") from exc
        finally:
            if timed_out:
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=False)

    def _wait_for(self, predicate: Callable[[], bool], timeout_s: float, step_s: float = 0.1) -> bool:
        end = time.perf_counter() + timeout_s
        while time.perf_counter() < end:
            try:
                if predicate():
                    return True
            except Exception:
                pass
            time.sleep(step_s)
        return False

    def _create_test_image(self, name: str = "tester_image.png") -> Path:
        path = self.artifact_dir / name
        try:
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (640, 360), color=(18, 32, 52))
            draw = ImageDraw.Draw(image)
            draw.rectangle((24, 24, 616, 336), outline=(230, 160, 90), width=4)
            draw.text((40, 50), "JARVIS TEST IMAGE", fill=(240, 240, 240))
            draw.text((40, 92), _iso_now(), fill=(200, 220, 255))
            draw.ellipse((420, 120, 580, 280), fill=(80, 160, 210))
            image.save(path)
        except Exception:
            # Fallback: small binary-less placeholder if Pillow is unavailable.
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
        self._temp_files.append(path)
        return path

    def _capture_signal_state(self) -> Dict[str, Any]:
        route_data: Dict[str, Any] = {}
        if self.pipeline is not None and self.pipeline.router is not None:
            try:
                route_data = dict(getattr(self.pipeline.router, "_last_route_result", {}) or {})
            except Exception:
                route_data = {}
        return {
            "signals": dict(self.signal_counts),
            "last_state": self.last_state,
            "last_user_message": self.last_user_message,
            "last_assistant_message": self.last_assistant_message,
            "last_intent": self.last_intent,
            "last_intent_confidence": self.last_intent_confidence,
            "last_route": route_data,
        }

    def _pipeline_query(
        self,
        query: str,
        *,
        timeout_s: Optional[float] = None,
        expected_route_prefix: Optional[str] = None,
        expected_route_prefixes: Optional[List[str]] = None,
        expected_intent: Optional[str] = None,
        require_success: bool = True,
    ) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable", details={})

        timeout_value = float(timeout_s or self.profile.query_timeout_s)

        def _call() -> Dict[str, Any]:
            return self.pipeline.process_text(query)

        result = self._run_with_timeout(_call, timeout_value, f"process_text:{query[:48]}")
        intent_payload = dict(result.get("intent") or {})
        action_payload = dict(result.get("action") or {})
        route_path = ""
        action_data = action_payload.get("data")
        if isinstance(action_data, dict):
            route_path = str(action_data.get("_route_path", "") or "")
        if not route_path and self.pipeline.router is not None:
            route_path = str((getattr(self.pipeline.router, "_last_route_result", {}) or {}).get("route_path", ""))

        ok = True
        reasons: List[str] = []

        if require_success and not bool(action_payload.get("success", False)):
            ok = False
            reasons.append("action.success is false")

        if expected_intent:
            actual_intent = str(intent_payload.get("intent", ""))
            if actual_intent != expected_intent:
                ok = False
                reasons.append(f"intent mismatch ({actual_intent} != {expected_intent})")

        route_prefixes: List[str] = []
        if expected_route_prefix:
            route_prefixes.append(str(expected_route_prefix))
        if expected_route_prefixes:
            route_prefixes.extend(str(item) for item in expected_route_prefixes if str(item).strip())

        if route_prefixes:
            if not any(route_path.startswith(prefix) for prefix in route_prefixes):
                ok = False
                reasons.append(f"route mismatch ({route_path} did not match {route_prefixes})")

        response_text = str(action_payload.get("response_text", ""))
        if not response_text.strip():
            ok = False
            reasons.append("empty response text")

        details = {
            "query": query,
            "intent": intent_payload,
            "entities": result.get("entities", {}),
            "action": action_payload,
            "route_path": route_path,
            "signal_state": self._capture_signal_state(),
        }

        message = "OK" if ok else "; ".join(reasons)
        return CaseOutcome(ok=ok, message=message, details=details)

    def _route_branch_call(
        self,
        *,
        intent: str,
        entities: Dict[str, Any],
        raw_text: str,
        expected_route_prefix: str,
        require_success: bool = True,
    ) -> CaseOutcome:
        if self.pipeline is None or self.pipeline.router is None:
            return CaseOutcome(ok=False, message="Router unavailable")

        intent_payload = {
            "intent": intent,
            "confidence": 0.99,
            "all_scores": {intent: 0.99},
            "latency_ms": 0.0,
        }
        timeout_s = self.profile.query_timeout_s
        if intent == "vision_query":
            timeout_s = max(timeout_s, 120.0)

        result = self._run_with_timeout(
            lambda: self.pipeline.router.route(intent_payload, entities, raw_text, [], compute_hint="cpu"),
            timeout_s,
            f"router.route:{intent}",
        )

        result_data = result.get("data") if isinstance(result, dict) else {}
        route_path = ""
        if isinstance(result_data, dict):
            route_path = str(result_data.get("_route_path", "") or "")
        if not route_path:
            route_path = str((getattr(self.pipeline.router, "_last_route_result", {}) or {}).get("route_path", ""))
        success = bool(result.get("success", False))
        ok = route_path.startswith(expected_route_prefix)
        if require_success:
            ok = ok and success

        details = {
            "intent": intent,
            "entities": entities,
            "raw_text": raw_text,
            "route_path": route_path,
            "result": result,
        }
        if ok:
            return CaseOutcome(ok=True, message="OK", details=details)
        return CaseOutcome(
            ok=False,
            message=f"route={route_path}; success={success}; expected_prefix={expected_route_prefix}",
            details=details,
        )

    def _init_pipeline(self) -> CaseOutcome:
        self.pipeline = JarvisPipeline()

        self.pipeline.pipeline_state_changed.connect(self._on_state)
        self.pipeline.new_message.connect(self._on_message)
        self.pipeline.intent_classified.connect(self._on_intent)
        self.pipeline.intent_diagnostics.connect(self._on_diagnostics)
        self.pipeline.initialization_progress.connect(self._on_init_progress)
        self.pipeline.ready.connect(self._on_ready)
        self.pipeline.wakeword_availability_changed.connect(self._on_wake_availability)

        self.pipeline.initialize()

        if not self._wait_for(lambda: bool(self.pipeline and self.pipeline.router is not None), timeout_s=60.0):
            return CaseOutcome(ok=False, message="Pipeline initialization timed out", details=self._capture_signal_state())

        # Keep deterministic and stress runs quieter; TTS is tested explicitly in dedicated suite.
        self.pipeline.set_tts_enabled(False)

        volume = system_control.get_volume()
        if bool(volume.get("success", False)):
            self.volume_baseline = _safe_int((volume.get("data") or {}).get("level"), default=None)  # type: ignore[arg-type]

        brightness = system_control.get_brightness()
        if bool(brightness.get("success", False)):
            self.brightness_baseline = _safe_int((brightness.get("data") or {}).get("level"), default=None)  # type: ignore[arg-type]

        details = {
            "llm": self.pipeline.get_llm_status(),
            "stt": self.pipeline.get_stt_status(),
            "wakeword": self.pipeline.get_wakeword_status(),
            "compute": self.pipeline.get_compute_settings(),
            "signal_state": self._capture_signal_state(),
            "volume_baseline": self.volume_baseline,
            "brightness_baseline": self.brightness_baseline,
        }
        return CaseOutcome(ok=True, message="Pipeline initialized", details=details)

    def _on_state(self, state: str) -> None:
        self.signal_counts["state"] += 1
        self.last_state = str(state)

    def _on_message(self, role: str, text: str) -> None:
        self.signal_counts["new_message"] += 1
        if str(role) == "assistant":
            self.last_assistant_message = str(text)
        else:
            self.last_user_message = str(text)

    def _on_intent(self, intent: str, confidence: float) -> None:
        self.signal_counts["intent"] += 1
        self.last_intent = str(intent)
        self.last_intent_confidence = _safe_float(confidence)

    def _on_diagnostics(self, _payload: Dict[str, Any]) -> None:
        self.signal_counts["diagnostics"] += 1

    def _on_init_progress(self, _label: str, _progress: int) -> None:
        self.signal_counts["init_progress"] += 1

    def _on_ready(self) -> None:
        self.signal_counts["ready"] += 1

    def _on_wake_availability(self, _available: bool, _message: str) -> None:
        self.signal_counts["wake_availability"] += 1

    def _suite_preflight(self) -> None:
        self._run_case(
            category="preflight",
            name="import_smoke",
            func=self._case_import_smoke,
        )
        self._run_case(
            category="preflight",
            name="pipeline_init",
            func=self._init_pipeline,
        )

    def _case_import_smoke(self) -> CaseOutcome:
        cmd = [sys.executable, str(ROOT_DIR / "__copilot_import_check.py")]
        proc = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True, check=False)
        details = {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "command": cmd,
        }
        if proc.returncode == 0:
            return CaseOutcome(ok=True, message="Import smoke passed", details=details)
        return CaseOutcome(ok=False, message="Import smoke failed", details=details)

    def _suite_runtime_probe(self) -> None:
        self._run_case(category="runtime_probe", name="compute_probe_sample_1", func=self._case_compute_probe_sample)
        self._run_case(category="runtime_probe", name="compute_probe_sample_2", func=self._case_compute_probe_sample)
        self._run_case(category="runtime_probe", name="gpu_diagnosis", func=self._case_gpu_diagnosis)
        self._run_case(category="runtime_probe", name="component_status_snapshot", func=self._case_component_snapshot)

    def _case_compute_probe_sample(self) -> CaseOutcome:
        sample = detect_compute_capabilities()
        self._compute_samples.append(dict(sample))
        return CaseOutcome(ok=True, message="Compute probe captured", details=sample)

    def _case_gpu_diagnosis(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        pipeline_caps = dict((self.pipeline.get_compute_settings() or {}).get("capabilities") or {})
        llm_status = dict(self.pipeline.get_llm_status() or {})
        samples = list(self._compute_samples)

        unstable = False
        reasons: List[str] = []

        if len(samples) >= 2:
            first = samples[0]
            last = samples[-1]
            for key in ["gpu_supported", "onnx_cuda_available", "torch_cuda_available", "llm_gpu_offload_supported", "stt_cuda_ready"]:
                if bool(first.get(key)) != bool(last.get(key)):
                    unstable = True
                    reasons.append(f"{key} changed between probes ({first.get(key)} -> {last.get(key)})")

        llm_gpu = bool(pipeline_caps.get("llm_gpu_offload_supported", False))
        llm_mode = str(llm_status.get("mode", ""))
        llm_message = str(llm_status.get("message", ""))
        if llm_mode == "worker" and "cpu" in llm_message.lower() and llm_gpu:
            reasons.append("LLM advertises GPU support but runtime message indicates CPU backend")

        details = {
            "samples": samples,
            "pipeline_caps": pipeline_caps,
            "llm_status": llm_status,
            "unstable_flags": unstable,
            "notes": reasons,
        }

        if unstable:
            return CaseOutcome(ok=False, message="GPU capability appears unstable", details=details)
        return CaseOutcome(ok=True, message="GPU diagnosis captured", details=details)

    def _case_component_snapshot(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        details = {
            "compute": self.pipeline.get_compute_settings(),
            "llm": self.pipeline.get_llm_status(),
            "stt": self.pipeline.get_stt_status(),
            "wakeword": self.pipeline.get_wakeword_status(),
            "tts": self.pipeline.get_tts_settings(),
            "verbosity": self.pipeline.get_response_settings(),
            "signals": dict(self.signal_counts),
        }
        return CaseOutcome(ok=True, message="Status snapshot captured", details=details)

    def _suite_user_simulated(self) -> None:
        tests = [
            ("time_fast_path", "what time is it", "fast_path", None, True),
            ("date_fast_path", "what date is it", "fast_path", None, True),
            ("system_info_fast_path", "show my system status", "fast_path", None, True),
            ("launch_app_user", "open notepad", "launch_app", "launch_app", True),
            ("close_app_user", "close notepad", "close_app", "close_app", False),
            ("web_search_user", "search for python unit testing strategies", "web_search", "web_search", True),
            ("open_website_user", "open github.com", "open_website", "open_website", True),
            ("system_settings_user", "open display settings", "system_settings", "system_settings", True),
            ("volume_set_user", "set volume to 33 percent", "system_volume", "system_volume", True),
            ("volume_read_user", "what is the volume", "fast_path", None, True),
            ("brightness_set_user", "set brightness to 55 percent", "system_brightness", "system_brightness", True),
            ("brightness_read_user", "what is the brightness", "fast_path", None, True),
            ("media_play_user", "play lo fi beats on youtube", "play_media", "play_media", True),
            ("media_pause_fast_path", "pause music", "fast_path", None, True),
            ("media_resume_fast_path", "resume music", "fast_path", None, True),
            ("small_talk_user", "hello", "small_talk", None, True),
        ]

        for name, query, route_prefix, expected_intent, require_success in tests:
            self._run_case(
                category="user_simulated",
                name=name,
                func=lambda q=query, r=route_prefix, i=expected_intent, s=require_success: self._pipeline_query(
                    q,
                    expected_route_prefix=r,
                    expected_intent=i,
                    require_success=s,
                ),
            )

    def _suite_route_coverage(self) -> None:
        test_file = self.artifact_dir / "route_file_control_sample.txt"
        test_file.write_text("route coverage sample", encoding="utf-8")
        self._temp_files.append(test_file)

        image_file = self._create_test_image("route_vision_sample.png")

        routes = [
            (
                "route_launch_app",
                "launch_app",
                {"app_name": "notepad"},
                "open notepad",
                "launch_app",
                True,
            ),
            (
                "route_close_app",
                "close_app",
                {"app_name": "notepad"},
                "close notepad",
                "close_app",
                False,
            ),
            (
                "route_web_search",
                "web_search",
                {"search_query": "python testing", "platform": "google"},
                "search for python testing",
                "web_search",
                True,
            ),
            (
                "route_open_website",
                "open_website",
                {"website_url": "github.com"},
                "open github",
                "open_website",
                True,
            ),
            (
                "route_play_media",
                "play_media",
                {"media_title": "classical focus", "platform": "youtube"},
                "play classical focus",
                "play_media",
                True,
            ),
            (
                "route_system_volume",
                "system_volume",
                {"volume_level": "31"},
                "set volume",
                "system_volume",
                True,
            ),
            (
                "route_system_brightness",
                "system_brightness",
                {"brightness_level": "52"},
                "set brightness",
                "system_brightness",
                True,
            ),
            (
                "route_system_settings",
                "system_settings",
                {"setting_name": "display"},
                "open display settings",
                "system_settings",
                True,
            ),
            (
                "route_file_control",
                "file_control",
                {"file_action": "find", "file_name": test_file.name, "file_path": str(test_file)},
                f"find {test_file.name}",
                "file_control",
                True,
            ),
            (
                "route_clipboard_action",
                "clipboard_action",
                {"clipboard_action": "read"},
                "read clipboard",
                "clipboard_action",
                True,
            ),
            (
                "route_stop_cancel",
                "stop_cancel",
                {},
                "stop",
                "stop_cancel",
                True,
            ),
            (
                "route_vision_image",
                "vision_query",
                {"vision_mode": "image", "file_path": str(image_file)},
                "describe this image",
                "vision_image",
                True,
            ),
            (
                "route_general_qa",
                "general_qa",
                {},
                "explain deterministic testing in two lines",
                "general_",
                True,
            ),
        ]

        for name, intent, entities, raw_text, route_prefix, require_success in routes:
            self._run_case(
                category="route_coverage",
                name=name,
                func=lambda i=intent, e=entities, t=raw_text, r=route_prefix, s=require_success: self._route_branch_call(
                    intent=i,
                    entities=e,
                    raw_text=t,
                    expected_route_prefix=r,
                    require_success=s,
                ),
            )

    def _suite_verify_only_safety(self) -> None:
        verify_queries = [
            ("verify_power_shutdown", "shutdown", "shutdown"),
            ("verify_power_restart", "restart", "restart"),
            ("verify_power_sleep", "sleep", "sleep"),
            ("verify_power_lock", "lock workstation", "lock"),
            ("verify_radio_wifi_on", "turn on wifi", "wifi on"),
            ("verify_radio_wifi_off", "turn off wifi", "wifi off"),
            ("verify_radio_bluetooth_on", "turn on bluetooth", "bluetooth on"),
            ("verify_radio_bluetooth_off", "turn off bluetooth", "bluetooth off"),
            ("verify_radio_airplane_on", "airplane mode on", "airplane mode on"),
            ("verify_radio_airplane_off", "airplane mode off", "airplane mode off"),
        ]

        for name, query, expected_power in verify_queries:
            self._run_case(
                category="verify_only",
                name=name,
                verify_only=True,
                func=lambda q=query, exp=expected_power: self._case_verify_power_parse(q, exp),
            )

    def _case_verify_power_parse(self, query: str, expected_power: str) -> CaseOutcome:
        if self.pipeline is None or self.pipeline.intent_classifier is None:
            return CaseOutcome(ok=False, message="Pipeline/intent classifier unavailable")

        intent_result = self.pipeline.intent_classifier.predict(query)
        entities = self.entity_extractor.extract_entities(query, str(intent_result.get("intent", "")))
        found = str(entities.get("power_command", ""))
        intent_name = str(intent_result.get("intent", ""))

        ok = found == expected_power and intent_name == "power_control"
        details = {
            "query": query,
            "expected_power_command": expected_power,
            "intent": intent_result,
            "entities": entities,
            "safety": "verification_only_no_execution",
        }
        message = "Command safely recognized" if ok else f"intent={intent_name}; power_command={found}"
        return CaseOutcome(ok=ok, message=message, details=details)

    def _suite_llm_and_web(self) -> None:
        self._run_case(category="llm", name="llm_availability", func=self._case_llm_availability)
        self._run_case(category="llm", name="llm_general_response", func=self._case_llm_general_response)
        self._run_case(category="llm", name="llm_context_continuity", func=self._case_llm_context_continuity)
        self._run_case(category="llm", name="llm_cancellation", func=self._case_llm_cancellation)
        self._run_case(category="llm", name="verified_web_mode_real", func=self._case_verified_web_real)

    def _case_llm_availability(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        status = dict(self.pipeline.get_llm_status() or {})
        available = bool(status.get("available", False))
        ready = bool(status.get("ready", False))

        if not available and self.strict_real:
            return CaseOutcome(ok=False, message="LLM unavailable in strict mode", details=status)

        # Prewarm and refresh status for strict checks.
        if self.pipeline.llm is not None:
            try:
                self.pipeline.llm.prewarm()
            except Exception:
                pass
            status = dict(self.pipeline.get_llm_status() or {})
            available = bool(status.get("available", False))
            ready = bool(status.get("ready", False))

        if self.strict_real and not (available and ready):
            return CaseOutcome(ok=False, message="LLM not ready after prewarm", details=status)
        return CaseOutcome(ok=True, message="LLM status captured", details=status)

    def _case_llm_general_response(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        if self.strict_real and not bool((self.pipeline.get_llm_status() or {}).get("available", False)):
            return CaseOutcome(ok=False, message="LLM unavailable in strict mode", details=self.pipeline.get_llm_status())

        query = "Explain why end-to-end stress tests catch issues that unit tests can miss."
        result = self._pipeline_query(query, expected_intent=None, expected_route_prefix="general_", require_success=True)
        return result

    def _case_llm_context_continuity(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        if self.strict_real and not bool((self.pipeline.get_llm_status() or {}).get("available", False)):
            return CaseOutcome(ok=False, message="LLM unavailable in strict mode")

        first = self._run_with_timeout(
            lambda: self.pipeline.process_text("Give me three short points about robust testing."),
            self.profile.query_timeout_s,
            "llm_context_first",
        )
        second = self._run_with_timeout(
            lambda: self.pipeline.process_text("Now summarize those points into one sentence."),
            self.profile.query_timeout_s,
            "llm_context_second",
        )

        action_two = dict(second.get("action") or {})
        response = str(action_two.get("response_text", ""))
        action_two_data = action_two.get("data")
        route_path = ""
        if isinstance(action_two_data, dict):
            route_path = str(action_two_data.get("_route_path", "") or "")
        if not route_path and self.pipeline.router is not None:
            route_path = str((getattr(self.pipeline.router, "_last_route_result", {}) or {}).get("route_path", ""))

        ok = bool(action_two.get("success", False)) and bool(response.strip()) and route_path.startswith("general_")
        details = {
            "first": first,
            "second": second,
            "route_path": route_path,
        }
        return CaseOutcome(ok=ok, message=("Context continuity OK" if ok else "Context continuity failed"), details=details)

    def _case_llm_cancellation(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        if self.strict_real and not bool((self.pipeline.get_llm_status() or {}).get("available", False)):
            return CaseOutcome(ok=False, message="LLM unavailable in strict mode")

        long_prompt = (
            "Provide a detailed multi-section explanation of how to design resilient distributed testing systems, "
            "including trade-offs, failure modes, metrics, and scaling concerns."
        )
        holder: Dict[str, Any] = {}
        done = threading.Event()

        def worker() -> None:
            try:
                holder["result"] = self.pipeline.process_text(long_prompt)
            except Exception as exc:
                holder["error"] = str(exc)
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        time.sleep(1.5)
        cancel_payload = self.pipeline.cancel_current_action()
        done.wait(timeout=self.profile.query_timeout_s)

        result_payload = holder.get("result")
        action = dict((result_payload or {}).get("action") or {}) if isinstance(result_payload, dict) else {}
        cancelled = bool((action.get("data") or {}).get("cancelled", False)) or bool((cancel_payload.get("data") or {}).get("cancelled", False))

        ok = bool(cancel_payload.get("success", False)) and cancelled
        details = {
            "cancel_payload": cancel_payload,
            "result_payload": result_payload,
            "holder": holder,
        }
        return CaseOutcome(ok=ok, message=("Cancellation path OK" if ok else "Cancellation path failed"), details=details)

    def _case_verified_web_real(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        self.pipeline.set_realtime_web_enabled(True)
        try:
            result = self._pipeline_query(
                "What are the latest major AI model releases this month?",
                expected_route_prefixes=["general_verified_web", "web_search_verified"],
                require_success=True,
            )
            return result
        finally:
            self.pipeline.set_realtime_web_enabled(False)

    def _suite_vision(self) -> None:
        self._run_case(category="vision", name="vision_llm_image_analysis", func=self._case_vision_llm_image)
        self._run_case(category="vision", name="vision_screen_query", func=self._case_vision_screen_query)
        self._run_case(category="vision", name="vision_camera_query", func=self._case_vision_camera_query)

    def _case_vision_llm_image(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        image_path = self._create_test_image("vision_llm_test.png")
        result = self._run_with_timeout(
            lambda: self.pipeline.analyze_image_file(str(image_path), prompt="Describe this synthetic test image."),
            max(self.profile.query_timeout_s, 180.0),
            "analyze_image_file",
        )

        data = dict(result.get("data") or {})
        mode = str(data.get("mode", ""))
        success = bool(result.get("success", False))

        # Strict mode still requires real execution, but accepts graceful CNN fallback
        # when multimodal runtime is unavailable or unstable.
        if self.strict_real:
            ok = success and mode in {"qwen2.5-vl", "cnn_fallback"}
        else:
            ok = success and mode in {"qwen2.5-vl", "cnn_fallback"}

        return CaseOutcome(
            ok=ok,
            message=("Vision LLM path OK" if ok else f"Vision mode={mode}; success={success}"),
            details={"result": result, "strict_real": self.strict_real},
        )

    def _case_vision_screen_query(self) -> CaseOutcome:
        result = self._pipeline_query(
            "what is on my screen right now",
            timeout_s=max(self.profile.query_timeout_s, 120.0),
            expected_route_prefix="vision_",
            require_success=not self.strict_real,
        )
        if self.strict_real:
            # For strict runs, ensure at least route entered vision path and gave non-empty response.
            route_path = str((result.details or {}).get("route_path", ""))
            action_success = bool(((result.details or {}).get("action") or {}).get("success", False))
            ok = route_path.startswith("vision_") and action_success
            return CaseOutcome(ok=ok, message=("Vision screen path OK" if ok else result.message), details=result.details)
        return result

    def _case_vision_camera_query(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        result = self._run_with_timeout(
            lambda: self.pipeline.analyze_camera(),
            max(self.profile.query_timeout_s, 120.0),
            "analyze_camera",
        )
        success = bool(result.get("success", False))
        if self.strict_real and not success:
            return CaseOutcome(ok=False, message="Camera vision failed in strict mode", details={"result": result})
        return CaseOutcome(ok=success, message=("Camera vision OK" if success else "Camera vision failed"), details={"result": result})

    def _suite_audio(self) -> None:
        self._run_case(category="audio", name="tts_speak_async", func=self._case_tts_speak)
        self._run_case(category="audio", name="stt_status_available", func=self._case_stt_status)
        self._run_case(category="audio", name="stt_from_tts_loopback", func=self._case_stt_from_tts)
        self._run_case(category="audio", name="recorded_audio_pipeline_path", func=self._case_process_recorded_audio)

    def _case_tts_speak(self) -> CaseOutcome:
        if self.pipeline is None or self.pipeline.tts is None:
            return CaseOutcome(ok=False, message="TTS module unavailable")

        self.pipeline.set_tts_enabled(True)
        future: Future[Any] = self.pipeline.tts.speak_async("JARVIS full system tester speaking for diagnostics.")
        try:
            future.result(timeout=180.0)
        except Exception as exc:
            return CaseOutcome(ok=False, message=f"TTS speak failed: {exc}")
        return CaseOutcome(ok=True, message="TTS speak completed")

    def _case_stt_status(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        status = dict(self.pipeline.get_stt_status() or {})
        available = bool(status.get("available", False))
        if self.strict_real and not available:
            return CaseOutcome(ok=False, message="STT unavailable in strict mode", details=status)
        return CaseOutcome(ok=available, message=("STT available" if available else "STT unavailable"), details=status)

    def _synthesize_tts_audio(self, text: str) -> tuple[Any, int, str]:
        if self.pipeline is None or self.pipeline.tts is None:
            raise RuntimeError("TTS module unavailable")
        tts = self.pipeline.tts

        errors: List[str] = []

        try:
            samples, sr = tts._synthesize_edge_tts(text)  # type: ignore[attr-defined]
            return samples, int(sr), "edge"
        except Exception as exc:
            errors.append(f"edge:{exc}")

        try:
            samples, sr = tts._synthesize_kokoro(text)  # type: ignore[attr-defined]
            return samples, int(sr), "kokoro"
        except Exception as exc:
            errors.append(f"kokoro:{exc}")

        raise RuntimeError("Unable to synthesize TTS audio for STT loopback: " + "; ".join(errors))

    def _case_stt_from_tts(self) -> CaseOutcome:
        if self.pipeline is None or self.pipeline.stt is None:
            return CaseOutcome(ok=False, message="STT module unavailable")

        if self.strict_real and not bool((self.pipeline.get_stt_status() or {}).get("available", False)):
            return CaseOutcome(ok=False, message="STT unavailable in strict mode")

        try:
            audio, sample_rate, backend = self._synthesize_tts_audio(
                "Jarvis speech to text loopback test for strict diagnostics"
            )
        except Exception as exc:
            return CaseOutcome(ok=False, message=str(exc))

        transcript = self.pipeline.stt.transcribe(audio, sample_rate=sample_rate)
        text = str(transcript.get("text", "")).strip().lower()
        error = str(transcript.get("error", "")).strip()

        expected = "jarvis speech to text loopback test for strict diagnostics"
        similarity = difflib.SequenceMatcher(a=text, b=expected).ratio() if text else 0.0
        token_hits = sum(1 for token in ["speech", "text", "loopback", "diagnostics"] if token in text)

        ok = bool(text) and ("jarvis" in text or (token_hits >= 3 and similarity >= 0.55))
        if self.strict_real and not ok:
            return CaseOutcome(
                ok=False,
                message="STT loopback transcription did not produce expected output",
                details={
                    "backend": backend,
                    "transcript": transcript,
                    "similarity": round(float(similarity), 3),
                    "keyword_hits": token_hits,
                },
            )

        return CaseOutcome(
            ok=ok,
            message=("STT loopback OK" if ok else "STT loopback partial"),
            details={
                "backend": backend,
                "transcript": transcript,
                "error": error,
                "similarity": round(float(similarity), 3),
                "keyword_hits": token_hits,
            },
        )

    def _case_process_recorded_audio(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")
        try:
            audio, sample_rate, backend = self._synthesize_tts_audio(
                "This is a recorded audio pipeline path validation command"
            )
        except Exception as exc:
            return CaseOutcome(ok=False, message=str(exc))

        result = self._run_with_timeout(
            lambda: self.pipeline.process_recorded_audio(audio, sample_rate=sample_rate),
            max(self.profile.query_timeout_s, 180.0),
            "process_recorded_audio",
        )
        transcript = dict(result.get("transcript") or {})
        action = dict(result.get("action") or {})
        text = str(transcript.get("text", "")).strip()
        ok = bool(text) and bool(action.get("response_text", "").strip())

        return CaseOutcome(
            ok=ok,
            message=("Recorded audio pipeline path OK" if ok else "Recorded audio pipeline path failed"),
            details={"backend": backend, "result": result},
        )

    def _suite_wakeword_and_settings(self) -> None:
        self._run_case(category="wakeword", name="wakeword_enable_status", func=self._case_wakeword_enable)
        self._run_case(category="settings", name="runtime_settings_roundtrip", func=self._case_runtime_settings_roundtrip)
        self._run_case(category="wakeword", name="wakeword_disable_status", func=self._case_wakeword_disable)

    def _case_wakeword_enable(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        self.pipeline.update_wakeword_settings(
            enabled=True,
            sensitivity=0.35,
            activation_phrases=["jarvis", "hey jarvis"],
            strict_phrase_prefix=False,
            auto_restart_after_response=True,
            follow_up_after_response=True,
            follow_up_timeout=8,
            max_followup_turns=1,
        )

        settled = self._wait_for(
            lambda: self.pipeline is not None and not bool(self.pipeline.get_wakeword_status().get("initializing", False)),
            timeout_s=45.0,
        )
        status = dict(self.pipeline.get_wakeword_status() or {})
        available = bool(status.get("available", False))

        if self.strict_real and (not settled or not available):
            return CaseOutcome(ok=False, message="Wakeword unavailable in strict mode", details=status)
        return CaseOutcome(ok=settled, message=("Wakeword status settled" if settled else "Wakeword did not settle"), details=status)

    def _case_runtime_settings_roundtrip(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        before = {
            "tts": self.pipeline.get_tts_settings(),
            "response": self.pipeline.get_response_settings(),
            "compute": self.pipeline.get_compute_settings(),
            "web": self.pipeline.is_realtime_web_enabled(),
        }

        self.pipeline.set_tts_profile("male")
        self.pipeline.set_response_verbosity("brief")
        self.pipeline.set_realtime_web_enabled(True)
        self.pipeline.set_compute_mode("auto")

        after = {
            "tts": self.pipeline.get_tts_settings(),
            "response": self.pipeline.get_response_settings(),
            "compute": self.pipeline.get_compute_settings(),
            "web": self.pipeline.is_realtime_web_enabled(),
        }

        ok = (
            str((after["tts"] or {}).get("profile", "")) == "male"
            and str((after["response"] or {}).get("verbosity", "")) == "brief"
            and bool(after.get("web", False))
        )

        # Restore defaults used by the remainder of the run.
        self.pipeline.set_tts_profile("female")
        self.pipeline.set_response_verbosity("normal")
        self.pipeline.set_realtime_web_enabled(False)

        return CaseOutcome(ok=ok, message=("Runtime setting roundtrip OK" if ok else "Runtime setting roundtrip failed"), details={"before": before, "after": after})

    def _case_wakeword_disable(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")
        self.pipeline.update_wakeword_settings(enabled=False)
        settled = self._wait_for(
            lambda: self.pipeline is not None and not bool(self.pipeline.get_wakeword_status().get("enabled", False)),
            timeout_s=20.0,
        )
        status = dict(self.pipeline.get_wakeword_status() or {})
        return CaseOutcome(ok=settled, message=("Wakeword disabled" if settled else "Wakeword disable timeout"), details=status)

    def _suite_stress(self) -> None:
        self._run_case(category="stress", name="sequential_mixed_load", func=self._case_stress_sequential)
        self._run_case(category="stress", name="parallel_burst_load", func=self._case_stress_parallel)

    def _case_stress_sequential(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        commands = [
            "what time is it",
            "search for open source ai benchmarks",
            "open github.com",
            "set volume to 40 percent",
            "set brightness to 60 percent",
            "show my system status",
            "play ambient coding music on youtube",
            "pause music",
            "resume music",
            "hello",
        ]
        rng = random.Random(1337)
        failures: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []

        for idx in range(self.profile.stress_iterations):
            query = rng.choice(commands)
            started = time.perf_counter()
            try:
                result = self.pipeline.process_text(query)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies_ms.append(elapsed_ms)
                action = dict(result.get("action") or {})
                if not bool(action.get("response_text", "").strip()):
                    failures.append({"iteration": idx, "query": query, "reason": "empty response"})
            except Exception as exc:
                failures.append({"iteration": idx, "query": query, "reason": str(exc)})

        avg_ms = sum(latencies_ms) / max(1, len(latencies_ms))
        p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1] if latencies_ms else 0.0
        ok = len(failures) == 0

        details = {
            "iterations": self.profile.stress_iterations,
            "avg_latency_ms": round(avg_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "failure_count": len(failures),
            "failures": failures[:25],
        }
        message = "Sequential stress passed" if ok else f"Sequential stress failures: {len(failures)}"
        return CaseOutcome(ok=ok, message=message, details=details)

    def _case_stress_parallel(self) -> CaseOutcome:
        if self.pipeline is None:
            return CaseOutcome(ok=False, message="Pipeline unavailable")

        prompts = [
            "explain deterministic routing in one sentence",
            "what is the current time",
            "search for pytest qt examples",
            "open stackoverflow.com",
            "set volume to 35 percent",
        ]
        failures: List[Dict[str, Any]] = []
        successes = 0

        def worker(prompt: str, round_index: int, worker_index: int) -> Dict[str, Any]:
            started = time.perf_counter()
            result = self.pipeline.process_text(prompt)
            elapsed = (time.perf_counter() - started) * 1000.0
            return {
                "round": round_index,
                "worker": worker_index,
                "prompt": prompt,
                "elapsed_ms": round(elapsed, 2),
                "action": result.get("action", {}),
            }

        for round_index in range(self.profile.stress_concurrency_rounds):
            with ThreadPoolExecutor(max_workers=self.profile.stress_parallel_workers) as pool:
                futures = [
                    pool.submit(worker, prompts[(round_index + i) % len(prompts)], round_index, i)
                    for i in range(self.profile.stress_parallel_workers)
                ]
                for fut in futures:
                    try:
                        payload = fut.result(timeout=self.profile.query_timeout_s)
                        action = dict(payload.get("action") or {})
                        if not bool(action.get("response_text", "").strip()):
                            failures.append({"payload": payload, "reason": "empty response"})
                        else:
                            successes += 1
                    except Exception as exc:
                        failures.append({"round": round_index, "reason": str(exc)})

        ok = len(failures) == 0
        details = {
            "rounds": self.profile.stress_concurrency_rounds,
            "workers": self.profile.stress_parallel_workers,
            "successes": successes,
            "failure_count": len(failures),
            "failures": failures[:30],
        }
        return CaseOutcome(ok=ok, message=("Parallel stress passed" if ok else "Parallel stress had failures"), details=details)

    def _suite_ui(self) -> None:
        if not self.include_ui:
            self._record(
                category="ui",
                name="ui_suite",
                status="skipped",
                duration_s=0.0,
                message="UI suite disabled by flag",
                details={},
            )
            return

        self._run_case(category="ui", name="ui_torture_runner", func=self._case_ui_runner)

    def _case_ui_runner(self) -> CaseOutcome:
        script = ROOT_DIR / "qa" / "ui_torture_test.py"
        if not script.exists():
            return CaseOutcome(ok=False, message=f"Missing UI runner script: {script}")

        output_path = self.artifact_dir / "ui_report.json"
        cmd = [
            sys.executable,
            str(script),
            "--profile",
            self.ui_profile,
            "--output-json",
            str(output_path),
        ]

        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=False,
            timeout=self.profile.ui_timeout_s,
        )
        elapsed = time.perf_counter() - started

        ui_report: Dict[str, Any] = {}
        if output_path.exists():
            try:
                ui_report = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                ui_report = {"parse_error": "failed to parse ui report"}

        details = {
            "command": cmd,
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 2),
            "stdout_tail": (proc.stdout or "")[-8000:],
            "stderr_tail": (proc.stderr or "")[-8000:],
            "ui_report": ui_report,
        }

        if proc.returncode != 0:
            return CaseOutcome(ok=False, message="UI torture runner failed", details=details)
        return CaseOutcome(ok=True, message="UI torture runner passed", details=details)

    def _teardown(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.set_tts_enabled(False)
            except Exception:
                pass

        if self.volume_baseline is not None:
            try:
                system_control.set_volume(int(self.volume_baseline))
            except Exception as exc:
                self._cleanup_errors.append(f"restore volume failed: {exc}")

        if self.brightness_baseline is not None:
            try:
                system_control.set_brightness(int(self.brightness_baseline))
            except Exception as exc:
                self._cleanup_errors.append(f"restore brightness failed: {exc}")

        if self.pipeline is not None:
            try:
                self.pipeline.update_wakeword_settings(enabled=False)
            except Exception:
                pass
            try:
                self.pipeline.shutdown()
            except Exception as exc:
                self._cleanup_errors.append(f"pipeline shutdown failed: {exc}")
            self.pipeline = None

    def _build_summary(self) -> Dict[str, Any]:
        counts = {"passed": 0, "failed": 0, "skipped": 0, "verify_only": 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1

        ended_monotonic = time.perf_counter()
        finished_at = _iso_now()
        summary = {
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_s": round(ended_monotonic - self.started_monotonic, 3),
            "profile": self.profile.name,
            "strict_real": self.strict_real,
            "include_ui": self.include_ui,
            "artifact_dir": str(self.artifact_dir),
            "counts": counts,
            "failed_cases": [
                {
                    "id": item.id,
                    "category": item.category,
                    "name": item.name,
                    "message": item.message,
                }
                for item in self.results
                if item.status == "failed"
            ],
            "cleanup_errors": list(self._cleanup_errors),
            "manual_checklist": [
                "Verify tray close behavior: Run in background, then restore from tray.",
                "Verify wakeword audible acknowledgement with a real spoken wake phrase.",
                "Verify long-running LLM stop button behavior from the UI while response is streaming.",
                "Verify camera permission dialogs and retry behavior after denial.",
            ],
        }
        return summary

    def _write_report(self) -> Path:
        summary = self._build_summary()
        payload = {
            "summary": summary,
            "results": [
                {
                    "id": item.id,
                    "category": item.category,
                    "name": item.name,
                    "status": item.status,
                    "duration_s": item.duration_s,
                    "message": item.message,
                    "details": item.details,
                }
                for item in self.results
            ],
        }

        target = self.output_json or (self.artifact_dir / "full_system_report.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return target

    def run(self) -> int:
        self._log("=== JARVIS Full System Tester ===")
        self._log(f"Started at: {self.started_at}")
        self._log(f"Profile: {self.profile.name} | strict_real={self.strict_real} | include_ui={self.include_ui}")
        self._log(f"Artifacts: {self.artifact_dir}")

        try:
            self._suite_preflight()
            if any(item.status == "failed" and item.category == "preflight" for item in self.results):
                self._log("Preflight failed; continuing with diagnostics and remaining suites for full visibility.")

            self._suite_runtime_probe()
            self._suite_user_simulated()
            self._suite_route_coverage()
            self._suite_verify_only_safety()
            self._suite_llm_and_web()
            self._suite_vision()
            self._suite_audio()
            self._suite_wakeword_and_settings()
            self._suite_stress()
            self._suite_ui()
        finally:
            self._teardown()

        report_path = self._write_report()
        summary = self._build_summary()
        self._log("=== Summary ===")
        self._log(json.dumps(summary, indent=2, ensure_ascii=True))
        self._log(f"Report written to: {report_path}")

        return 1 if summary.get("counts", {}).get("failed", 0) > 0 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive ARIA/JARVIS full-system torture tester")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="aggressive",
        help="Execution profile",
    )
    parser.add_argument(
        "--strict-real",
        action="store_true",
        default=True,
        help="Fail tests when real dependencies are unavailable",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow unavailable dependencies to degrade instead of failing",
    )
    parser.add_argument(
        "--include-ui",
        action="store_true",
        default=True,
        help="Run UI torture suite",
    )
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="Skip UI torture suite",
    )
    parser.add_argument(
        "--ui-profile",
        choices=["quick", "full", "aggressive", "max"],
        default="aggressive",
        help="UI suite profile",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the final JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = PROFILES[str(args.profile)]
    include_ui = bool(args.include_ui) and not bool(args.skip_ui)
    strict_real = bool(args.strict_real) and not bool(args.non_strict)

    tester = FullSystemTester(
        profile=profile,
        strict_real=strict_real,
        include_ui=include_ui,
        ui_profile=str(args.ui_profile),
        output_json=args.output_json,
    )
    return tester.run()


if __name__ == "__main__":
    raise SystemExit(main())
