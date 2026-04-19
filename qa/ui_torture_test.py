from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QColor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QCheckBox, QDialog, QDialogButtonBox, QMessageBox

from ui.main_window import JarvisMainWindow


@dataclass
class UIProfile:
    name: str
    ready_timeout_ms: int
    request_timeout_ms: int
    text_loops: int
    attachment_loops: int
    mic_loops: int


PROFILES: Dict[str, UIProfile] = {
    "quick": UIProfile(
        name="quick",
        ready_timeout_ms=120000,
        request_timeout_ms=45000,
        text_loops=4,
        attachment_loops=2,
        mic_loops=1,
    ),
    "full": UIProfile(
        name="full",
        ready_timeout_ms=180000,
        request_timeout_ms=65000,
        text_loops=10,
        attachment_loops=5,
        mic_loops=3,
    ),
    "aggressive": UIProfile(
        name="aggressive",
        ready_timeout_ms=300000,
        request_timeout_ms=85000,
        text_loops=24,
        attachment_loops=12,
        mic_loops=6,
    ),
    "max": UIProfile(
        name="max",
        ready_timeout_ms=420000,
        request_timeout_ms=120000,
        text_loops=50,
        attachment_loops=24,
        mic_loops=12,
    ),
}


@dataclass
class StepResult:
    id: int
    name: str
    status: str
    duration_s: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class UITortureRunner:
    def __init__(self, profile: UIProfile) -> None:
        self.profile = profile
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.steps: List[StepResult] = []
        self._next_id = 1

        self._temp_dir = Path(tempfile.mkdtemp(prefix="jarvis-ui-torture-"))
        self._temp_files: List[Path] = []

        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = JarvisMainWindow()

    def _record(self, name: str, status: str, duration_s: float, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        step = StepResult(
            id=self._next_id,
            name=name,
            status=status,
            duration_s=round(float(duration_s), 4),
            message=str(message),
            details=details or {},
        )
        self._next_id += 1
        self.steps.append(step)

        marker = {
            "passed": "PASS",
            "failed": "FAIL",
            "skipped": "SKIP",
        }.get(status, status.upper())
        print(f"[{marker}] {name} ({step.duration_s:.2f}s) -> {message}", flush=True)

    def _wait_for(self, predicate: Callable[[], bool], timeout_ms: int, poll_ms: int = 50) -> bool:
        deadline = time.perf_counter() + (timeout_ms / 1000.0)
        while time.perf_counter() < deadline:
            self.app.processEvents()
            try:
                if predicate():
                    return True
            except Exception:
                pass
            QTest.qWait(poll_ms)
        return False

    def _run_step(self, name: str, func: Callable[[], tuple[bool, str, Dict[str, Any]]]) -> None:
        started = time.perf_counter()
        try:
            ok, message, details = func()
            self._record(name, "passed" if ok else "failed", time.perf_counter() - started, message, details)
        except Exception as exc:
            self._record(
                name,
                "failed",
                time.perf_counter() - started,
                f"Unhandled exception: {exc}",
                {"traceback": traceback.format_exc()},
            )

    def _create_temp_image(self, idx: int) -> Path:
        path = self._temp_dir / f"ui_attach_{idx:03d}.png"
        image = QImage(480, 320, QImage.Format.Format_RGB32)
        image.fill(QColor(20 + (idx * 7) % 80, 30 + (idx * 5) % 120, 70 + (idx * 3) % 150))
        image.save(str(path))
        self._temp_files.append(path)
        return path

    def _step_startup_ready(self) -> tuple[bool, str, Dict[str, Any]]:
        self.window.show()
        self.window.start_pipeline()

        ready = self._wait_for(lambda: self.window.input_box.isEnabled(), timeout_ms=self.profile.ready_timeout_ms)
        details = {
            "input_enabled": self.window.input_box.isEnabled(),
            "send_enabled": self.window.send_btn.isEnabled(),
            "mic_enabled": self.window.mic_btn.isEnabled(),
            "plus_enabled": self.window.plus_btn.isEnabled(),
            "log_lines": len(getattr(self.window, "_logs", [])),
        }
        if not ready:
            return False, "UI did not become ready in time", details

        # Reduce prolonged audio playback during UI loops; TTS is tested in backend suite.
        if self.window.pipeline is not None:
            self.window.pipeline.set_tts_enabled(False)

        return True, "UI startup ready", details

    def _step_text_interaction_stress(self) -> tuple[bool, str, Dict[str, Any]]:
        prompts = [
            "what time is it",
            "open github.com",
            "search for pyqt testing",
            "set volume to 30 percent",
            "hello",
            "show my system status",
        ]
        failures: List[Dict[str, Any]] = []

        for i in range(self.profile.text_loops):
            prompt = prompts[i % len(prompts)]
            self.window.input_box.setText(prompt)
            QTest.keyClick(self.window.input_box, Qt.Key.Key_Return)

            done = self._wait_for(
                lambda: not bool(getattr(self.window, "_text_task_running", False)),
                timeout_ms=self.profile.request_timeout_ms,
            )
            if not done:
                failures.append({"iteration": i, "prompt": prompt, "reason": "text task timeout"})
                continue

            send_mode = bool(getattr(self.window, "_send_button_stop_mode", False))
            send_enabled = self.window.send_btn.isEnabled()
            mic_enabled = self.window.mic_btn.isEnabled()
            if send_mode or not send_enabled or not mic_enabled:
                failures.append(
                    {
                        "iteration": i,
                        "prompt": prompt,
                        "reason": "button state invalid after request",
                        "send_mode": send_mode,
                        "send_enabled": send_enabled,
                        "mic_enabled": mic_enabled,
                    }
                )

        ok = len(failures) == 0
        details = {
            "loops": self.profile.text_loops,
            "failure_count": len(failures),
            "failures": failures[:30],
            "chat_messages": int(self.window.chat.layout.count()),
        }
        return ok, ("Text interaction stress OK" if ok else "Text interaction stress failures"), details

    def _step_stop_button_flow(self) -> tuple[bool, str, Dict[str, Any]]:
        self.window.input_box.setText(
            "Provide a detailed multi-section explanation of distributed systems resiliency patterns and trade-offs"
        )
        self.window.send_btn.click()

        entered_stop_mode = self._wait_for(
            lambda: bool(getattr(self.window, "_send_button_stop_mode", False)),
            timeout_ms=5000,
        )
        if not entered_stop_mode:
            return False, "Send button did not enter stop mode", {}

        self.window.send_btn.click()
        recovered = self._wait_for(
            lambda: not bool(getattr(self.window, "_send_button_stop_mode", False))
            and self.window.send_btn.isEnabled(),
            timeout_ms=30000,
        )
        details = {
            "recovered": recovered,
            "text_task_running": bool(getattr(self.window, "_text_task_running", False)),
            "voice_task_running": bool(getattr(self.window, "_voice_task_running", False)),
        }
        return recovered, ("Stop button flow OK" if recovered else "Stop button flow did not recover"), details

    def _step_attachment_stress(self) -> tuple[bool, str, Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        for i in range(self.profile.attachment_loops):
            image_path = self._create_temp_image(i)
            self.window._attach_image(str(image_path), source="drive")
            QTest.qWait(40)
            if not self.window.attachment_bar.isVisible():
                failures.append({"iteration": i, "reason": "attachment bar not visible"})
                continue

            if self.window._attached_image_path is None:
                failures.append({"iteration": i, "reason": "attached image path missing"})
                continue

            self.window._remove_attached_image()
            QTest.qWait(40)
            if self.window._attached_image_path is not None or self.window.attachment_bar.isVisible():
                failures.append({"iteration": i, "reason": "attachment did not clear"})

        ok = len(failures) == 0
        details = {
            "loops": self.profile.attachment_loops,
            "failure_count": len(failures),
            "failures": failures,
        }
        return ok, ("Attachment stress OK" if ok else "Attachment stress failures"), details

    def _step_hold_to_talk_stress(self) -> tuple[bool, str, Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        for i in range(self.profile.mic_loops):
            self.window._start_hold_to_talk()
            QTest.qWait(650)
            self.window._stop_hold_to_talk()

            settled = self._wait_for(
                lambda: not bool(getattr(self.window, "_voice_capture_active", False))
                and not bool(getattr(self.window, "_voice_task_running", False)),
                timeout_ms=max(30000, self.profile.request_timeout_ms),
            )
            if not settled:
                failures.append({"iteration": i, "reason": "voice capture/task did not settle"})
                continue

            if not self.window.mic_btn.isEnabled():
                failures.append({"iteration": i, "reason": "mic button stayed disabled"})

        ok = len(failures) == 0
        details = {
            "loops": self.profile.mic_loops,
            "failure_count": len(failures),
            "failures": failures,
        }
        return ok, ("Hold-to-talk stress OK" if ok else "Hold-to-talk stress failures"), details

    def _step_settings_dialog_apply(self) -> tuple[bool, str, Dict[str, Any]]:
        seen: Dict[str, Any] = {"dialog_found": False, "applied": False, "errors": []}

        def _interact() -> None:
            try:
                for widget in self.app.topLevelWidgets():
                    if isinstance(widget, QDialog) and widget.windowTitle() == "JARVIS Settings":
                        seen["dialog_found"] = True
                        checkboxes = widget.findChildren(QCheckBox)
                        if checkboxes:
                            original = checkboxes[0].isChecked()
                            checkboxes[0].setChecked(not original)
                            checkboxes[0].setChecked(original)
                        button_boxes = widget.findChildren(QDialogButtonBox)
                        if button_boxes:
                            ok_btn = button_boxes[0].button(QDialogButtonBox.StandardButton.Ok)
                            if ok_btn is not None:
                                ok_btn.click()
                                seen["applied"] = True
                                return
                        widget.accept()
                        seen["applied"] = True
                        return
            except Exception as exc:
                seen["errors"].append(str(exc))

        QTimer.singleShot(250, _interact)
        self.window._open_settings()

        ok = bool(seen.get("dialog_found")) and bool(seen.get("applied")) and not seen.get("errors")
        return ok, ("Settings dialog apply OK" if ok else "Settings dialog automation failed"), seen

    def _step_logs_dialog(self) -> tuple[bool, str, Dict[str, Any]]:
        for i in range(540):
            self.window._append_log(f"ui torture log line {i}")

        size_ok = len(getattr(self.window, "_logs", [])) <= 500
        seen = {"dialog_found": False, "closed": False}

        def _close_logs_dialog() -> None:
            for widget in self.app.topLevelWidgets():
                if isinstance(widget, QDialog) and widget.windowTitle() == "JARVIS Logs":
                    seen["dialog_found"] = True
                    widget.accept()
                    seen["closed"] = True
                    return

        QTimer.singleShot(200, _close_logs_dialog)
        self.window._open_logs()

        ok = size_ok and seen["dialog_found"] and seen["closed"]
        details = {
            "log_size": len(getattr(self.window, "_logs", [])),
            "dialog": seen,
        }
        return ok, ("Logs dialog path OK" if ok else "Logs dialog path failed"), details

    def _step_tray_toggle(self) -> tuple[bool, str, Dict[str, Any]]:
        if self.window.pipeline is None:
            return False, "Pipeline unavailable", {}

        try:
            self.window.tray.toggle_wake_word()
            QTest.qWait(300)
            self.window.tray.toggle_wake_word()
            QTest.qWait(300)
        except Exception as exc:
            return False, f"Tray toggle failed: {exc}", {}

        status = dict(self.window.pipeline.get_wakeword_status() or {})
        return True, "Tray wake toggle executed", {"wakeword_status": status, "tray_enabled": bool(self.window.tray.wake_enabled)}

    def _teardown(self) -> None:
        try:
            if self.window.pipeline is not None:
                self.window.pipeline.update_wakeword_settings(enabled=False)
                self.window.pipeline.set_tts_enabled(False)
        except Exception:
            pass

        try:
            self.window.tray.hide()
        except Exception:
            pass

        try:
            self.window._shutdown_for_exit()
        except Exception:
            pass

        try:
            self.window.close()
        except Exception:
            pass

        self.app.processEvents()

    def build_report(self) -> Dict[str, Any]:
        counts = {"passed": 0, "failed": 0, "skipped": 0}
        for step in self.steps:
            counts[step.status] = counts.get(step.status, 0) + 1
        return {
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "profile": self.profile.name,
            "counts": counts,
            "steps": [
                {
                    "id": item.id,
                    "name": item.name,
                    "status": item.status,
                    "duration_s": item.duration_s,
                    "message": item.message,
                    "details": item.details,
                }
                for item in self.steps
            ],
            "temp_dir": str(self._temp_dir),
        }

    def run(self) -> int:
        print("=== JARVIS UI Torture Runner ===", flush=True)
        print(f"Profile: {self.profile.name}", flush=True)

        try:
            self._run_step("startup_ready", self._step_startup_ready)
            self._run_step("text_interaction_stress", self._step_text_interaction_stress)
            self._run_step("stop_button_flow", self._step_stop_button_flow)
            self._run_step("attachment_stress", self._step_attachment_stress)
            self._run_step("hold_to_talk_stress", self._step_hold_to_talk_stress)
            self._run_step("settings_dialog_apply", self._step_settings_dialog_apply)
            self._run_step("logs_dialog", self._step_logs_dialog)
            self._run_step("tray_toggle", self._step_tray_toggle)
        finally:
            self._teardown()

        report = self.build_report()
        failures = report.get("counts", {}).get("failed", 0)
        print(json.dumps(report, indent=2, ensure_ascii=True), flush=True)
        return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UI torture runner for JARVIS")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="aggressive")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = PROFILES[str(args.profile)]
    runner = UITortureRunner(profile=profile)
    exit_code = runner.run()

    if args.output_json is not None:
        report = runner.build_report()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
