from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QApplication,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
)

from core.compute_runtime import estimate_query_complexity
from core.config import APP_ICON_PATH
from core.pipeline import JarvisPipeline
from core.session_logging import trace_event, trace_exception
from ui.system_tray import JarvisSystemTray
from ui.theme import BG_MAIN, BORDER, FONT_BODY, FONT_MONO, load_fonts
from ui.widgets.chat_widget import ChatWidget
from ui.widgets.orb_widget import OrbWidget
from ui.widgets.sidebar import Sidebar
from ui.widgets.waveform_widget import WaveformWidget
from speech.wakeword_config import DEFAULT_WAKEWORD_CONFIG, WAKEWORD_CONFIG_PATH
from vision.screen_capture import ScreenCapture
from vision.webcam import WebcamCapture


class RoundedContainer(QWidget):
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)

        painter.setBrush(BG_MAIN)
        painter.setPen(QPen(BORDER, 1))
        painter.drawRoundedRect(rect, 12, 12)
        super().paintEvent(event)


class JarvisMainWindow(QMainWindow):
    log_requested = pyqtSignal(str)
    voice_capture_failed = pyqtSignal(str)
    voice_capture_finished = pyqtSignal()
    text_task_finished = pyqtSignal()
    voice_task_finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        load_fonts()

        self.pipeline: JarvisPipeline | None = None
        self._logs: list[str] = []
        self._text_task_running = False
        self._voice_task_running = False
        self._voice_capture_active = False
        self._voice_capture_stop = threading.Event()
        self._voice_capture_chunks: list[np.ndarray] = []
        self._voice_sample_rate = 16000
        self._voice_capture_started_at = 0.0
        self._send_button_stop_mode = False
        self._long_task_hint_shown = False
        self._long_task_hint_text = "Processing, this may take a while..."
        self._attached_image_path: Path | None = None
        self._attached_image_source = ""
        self._last_image_path: Path | None = None
        self._last_image_source = ""
        self._resume_wake_after_hold = False
        self._is_shutting_down = False
        self._screen_capture = ScreenCapture()
        self._webcam_capture = WebcamCapture()

        self.setWindowTitle("JARVIS - Just A Really Very Intelligent System")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1180, 740)
        self.setMinimumSize(920, 620)
        self._center_on_screen()

        self.root = RoundedContainer(self)
        self.setCentralWidget(self.root)

        self.outer_layout = QVBoxLayout(self.root)
        self.outer_layout.setContentsMargins(12, 12, 12, 12)
        self.outer_layout.setSpacing(8)

        content = QHBoxLayout()
        content.setSpacing(12)
        self.outer_layout.addLayout(content, 1)

        self.sidebar = Sidebar(self.root)
        content.addWidget(self.sidebar, 0)

        center_panel = QWidget(self.root)
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 6, 10, 6)
        center_layout.setSpacing(10)

        self.orb = OrbWidget(center_panel)
        center_layout.addWidget(self.orb, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.llm_status_label = QLabel("", center_panel)
        self.llm_status_label.setFont(FONT_MONO)
        self.llm_status_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.llm_status_label.setWordWrap(True)
        self.llm_status_label.setStyleSheet("color: rgba(180, 220, 255, 220);")
        self._last_persistent_llm_status = ""
        self.llm_status_label.setVisible(False)
        center_layout.addWidget(self.llm_status_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.processing_hint_label = QLabel("", center_panel)
        self.processing_hint_label.setFont(FONT_MONO)
        self.processing_hint_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.processing_hint_label.setWordWrap(True)
        self.processing_hint_label.setStyleSheet("color: rgba(255, 190, 140, 220);")
        self.processing_hint_label.setVisible(False)
        center_layout.addWidget(self.processing_hint_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.waveform = WaveformWidget(center_panel)
        center_layout.addWidget(self.waveform)

        self.plus_btn = QToolButton(center_panel)
        self.plus_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self.plus_btn.setToolTip("Attach image or toggle extras")
        self.plus_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.plus_btn.setFixedWidth(38)

        self.plus_menu = QMenu(self.plus_btn)
        self.verified_web_action = QAction("Verified Web Mode", self.plus_menu)
        self.verified_web_action.setCheckable(True)
        self.verified_web_action.toggled.connect(self._on_verified_web_toggled)
        self.plus_menu.addAction(self.verified_web_action)
        self.plus_menu.addSeparator()

        self.analyze_file_action = QAction("Attach Image", self.plus_menu)
        self.analyze_file_action.triggered.connect(self._analyze_image_from_file)
        self.plus_menu.addAction(self.analyze_file_action)

        self.attach_screenshot_action = QAction("Attach Screenshot", self.plus_menu)
        self.attach_screenshot_action.triggered.connect(self._attach_screenshot)
        self.plus_menu.addAction(self.attach_screenshot_action)

        self.attach_camera_action = QAction("Attach Camera Snapshot", self.plus_menu)
        self.attach_camera_action.triggered.connect(self._attach_camera_snapshot)
        self.plus_menu.addAction(self.attach_camera_action)

        self.plus_menu.addSeparator()

        self.view_attached_action = QAction("View Last Image", self.plus_menu)
        self.view_attached_action.triggered.connect(self._view_last_image)
        self.plus_menu.addAction(self.view_attached_action)

        self.remove_attached_action = QAction("Remove Attached Image", self.plus_menu)
        self.remove_attached_action.triggered.connect(self._remove_attached_image)
        self.plus_menu.addAction(self.remove_attached_action)

        self.plus_btn.setMenu(self.plus_menu)
        self.plus_btn.setEnabled(False)

        self.attachment_bar = QFrame(center_panel)
        attachment_layout = QHBoxLayout(self.attachment_bar)
        attachment_layout.setContentsMargins(8, 6, 8, 6)
        attachment_layout.setSpacing(8)

        self.attachment_thumb = QLabel(self.attachment_bar)
        self.attachment_thumb.setFixedSize(54, 54)
        self.attachment_thumb.setScaledContents(False)
        self.attachment_thumb.setToolTip("Click to preview last image")
        self.attachment_thumb.mousePressEvent = self._on_attachment_thumb_clicked
        attachment_layout.addWidget(self.attachment_thumb)

        self.attachment_label = QLabel("No image attached", self.attachment_bar)
        self.attachment_label.setWordWrap(True)
        self.attachment_label.setFont(FONT_MONO)
        attachment_layout.addWidget(self.attachment_label, 1)

        self.attachment_remove_btn = QToolButton(self.attachment_bar)
        self.attachment_remove_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton))
        self.attachment_remove_btn.setToolTip("Remove attached image")
        self.attachment_remove_btn.clicked.connect(self._remove_attached_image)
        attachment_layout.addWidget(self.attachment_remove_btn)
        self.attachment_bar.setVisible(False)
        center_layout.addWidget(self.attachment_bar)

        input_row = QHBoxLayout()

        self.input_box = QLineEdit(center_panel)
        self.input_box.setPlaceholderText("Ask JARVIS anything...")
        self.input_box.returnPressed.connect(self._submit_text)
        self.input_box.setEnabled(False)

        self.send_btn = QPushButton("", center_panel)
        self.send_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.send_btn.setToolTip("Send")
        self.send_btn.setFixedWidth(42)
        self.send_btn.setFont(FONT_BODY)
        self.send_btn.clicked.connect(self._on_send_or_stop_clicked)
        self.send_btn.setEnabled(False)

        self.inline_attachment_icon = QLabel(center_panel)
        self.inline_attachment_icon.setFixedSize(22, 22)
        self.inline_attachment_icon.setVisible(False)
        self.inline_attachment_icon.setToolTip("Attached image")

        self.mic_btn = QPushButton("", center_panel)
        self.mic_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume))
        self.mic_btn.setToolTip("Hold to talk")
        self.mic_btn.setFixedWidth(42)
        self.mic_btn.setFont(FONT_BODY)
        self.mic_btn.pressed.connect(self._start_hold_to_talk)
        self.mic_btn.released.connect(self._stop_hold_to_talk)
        self.mic_btn.setEnabled(False)

        input_row.addWidget(self.plus_btn)
        input_row.addWidget(self.input_box, 1)
        input_row.addWidget(self.inline_attachment_icon)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.mic_btn)
        center_layout.addLayout(input_row)

        content.addWidget(center_panel, 1)

        right_panel = QWidget(self.root)
        right_panel.setMinimumWidth(250)
        right_panel.setMaximumWidth(540)
        right_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 6, 4, 6)
        right_layout.setSpacing(8)

        chat_title = QLabel("Conversation", right_panel)
        chat_title.setFont(FONT_MONO)
        right_layout.addWidget(chat_title)

        self.chat = ChatWidget(right_panel)
        right_layout.addWidget(self.chat, 1)

        content.addWidget(right_panel, 0)
        content.setStretch(0, 2)
        content.setStretch(1, 6)
        content.setStretch(2, 3)

        self.tray = JarvisSystemTray(self, APP_ICON_PATH, self)
        self.tray.show()

        self.sidebar.settings_btn.clicked.connect(self._open_settings)
        self.sidebar.logs_btn.clicked.connect(self._open_logs)
        self.sidebar.model_selector.currentTextChanged.connect(self._on_intent_model_changed)
        self.log_requested.connect(self._append_log)
        self.voice_capture_failed.connect(self._on_hold_capture_failed)
        self.voice_capture_finished.connect(self._on_hold_capture_finished)
        self.text_task_finished.connect(self._on_text_task_complete)
        self.voice_task_finished.connect(self._on_voice_task_complete)
        self._long_task_hint_timer = QTimer(self)
        self._long_task_hint_timer.setSingleShot(True)
        self._long_task_hint_timer.timeout.connect(self._on_long_task_hint_timeout)
        self._refresh_attachment_actions()
        self._trace(
            "initialized",
            width=self.width(),
            height=self.height(),
            wakeword_config_path=str(WAKEWORD_CONFIG_PATH),
        )

    def _trace(self, event: str, **details) -> None:
        trace_event("ui.main_window", event, **details)

    def _start_long_task_hint_timer(self, delay_ms: int = 2400, hint_text: str = "Processing, this may take a while...") -> None:
        self._long_task_hint_shown = False
        self._long_task_hint_text = str(hint_text or "Processing, this may take a while...").strip() or "Processing, this may take a while..."
        self._set_processing_hint_visible(False)
        self._long_task_hint_timer.start(max(800, int(delay_ms)))

    def _stop_long_task_hint_timer(self) -> None:
        if self._long_task_hint_timer.isActive():
            self._long_task_hint_timer.stop()
        self._long_task_hint_shown = False
        self._set_processing_hint_visible(False)

    def _set_processing_hint_visible(self, visible: bool, text: str = "") -> None:
        if visible:
            self.processing_hint_label.setText(text)
            self.processing_hint_label.setVisible(True)
            return

        self.processing_hint_label.clear()
        self.processing_hint_label.setVisible(False)

    def _should_hint_for_request(self, text: str, *, has_image: bool = False, from_voice: bool = False) -> tuple[bool, int]:
        if has_image:
            return True, 1000

        normalized = " ".join(str(text or "").strip().lower().split())
        if normalized and self.pipeline is not None and self.pipeline.is_realtime_web_enabled():
            try:
                if "web" in normalized or "online" in normalized or "latest" in normalized or "news" in normalized:
                    from actions import realtime_web

                    if realtime_web.looks_like_research_query(normalized):
                        return True, 1400
            except Exception:
                pass

        if normalized:
            try:
                complexity = estimate_query_complexity(normalized)
                if bool(complexity.get("is_complex")) and int(complexity.get("token_count", 0)) >= 20:
                    return True, 3800
            except Exception:
                pass

        if from_voice:
            return True, 6500

        return False, 0

    def _configure_processing_hint(self, text: str, *, has_image: bool = False, from_voice: bool = False) -> None:
        should_show, delay_ms = self._should_hint_for_request(text, has_image=has_image, from_voice=from_voice)
        if not should_show:
            self._stop_long_task_hint_timer()
            return
        self._start_long_task_hint_timer(delay_ms=delay_ms)

    def _on_long_task_hint_timeout(self) -> None:
        if self._long_task_hint_shown:
            return
        if not (self._text_task_running or self._voice_task_running):
            return
        self._long_task_hint_shown = True
        self._set_processing_hint_visible(True, self._long_task_hint_text)

    def _center_on_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + (geo.height() - self.height()) // 2
        self.move(x, y)

    def start_pipeline(self) -> None:
        selected_model = self.sidebar.model_selector.currentText()
        self._trace("pipeline_start_requested", intent_model=selected_model)
        self.pipeline = JarvisPipeline()
        self.pipeline.set_intent_model(selected_model)
        self.pipeline.pipeline_state_changed.connect(self._on_state_changed)
        self.pipeline.new_message.connect(self._on_new_message)
        self.pipeline.intent_classified.connect(self._on_intent)
        self.pipeline.intent_diagnostics.connect(self._on_intent_diagnostics)
        self.pipeline.initialization_progress.connect(self._on_init_progress)
        self.pipeline.wakeword_availability_changed.connect(self._on_wakeword_availability_changed)
        self.pipeline.ready.connect(self._on_pipeline_ready)
        self.pipeline.initialize_async()
        self._append_log(f"Intent model selected: {selected_model}")

    def _on_pipeline_ready(self) -> None:
        wake_available = False
        wake_enabled = False
        wake_initializing = False
        wake_reason = ""
        if self.pipeline:
            wake_status = self.pipeline.get_wakeword_status()
            wake_enabled = bool(wake_status.get("enabled", False))
            wake_available = bool(wake_status.get("available", False))
            wake_initializing = bool(wake_status.get("initializing", False))
            wake_reason = str(wake_status.get("reason", "") or "")
            if wake_enabled:
                self.pipeline.start_wake_word()

        self.tray.wake_enabled = wake_available
        self.input_box.setEnabled(True)
        self.send_btn.setEnabled(True)
        self._set_send_button_mode(False)
        self.mic_btn.setEnabled(True)
        self.plus_btn.setEnabled(True)
        self.verified_web_action.setChecked(self.pipeline.is_realtime_web_enabled())
        self._refresh_attachment_actions()
        self._append_log("JARVIS is online and ready.")
        self._trace(
            "pipeline_ready",
            wake_enabled=wake_enabled,
            wake_available=wake_available,
            wake_initializing=wake_initializing,
            realtime_web_enabled=bool(self.pipeline.is_realtime_web_enabled()),
        )

        if wake_enabled and wake_available:
            self._append_log("Wake-word listening is active in the background.")
        elif wake_enabled and wake_initializing:
            self._append_log("Wake-word listener is initializing in the background.")
        elif wake_enabled and not wake_available:
            self._append_log(wake_reason or "Wake-word listener is unavailable. You can still use hold-to-talk.")
        else:
            self._append_log("Wake-word listening is disabled by default for stability. Use Mic button for voice input.")

    def _on_wakeword_availability_changed(self, available: bool, message: str) -> None:
        if self.pipeline is None:
            return

        wake_status = self.pipeline.get_wakeword_status()
        wake_enabled = bool(wake_status.get("enabled", False))

        if wake_enabled:
            self.tray.wake_enabled = bool(available)
            if available:
                self.tray.setToolTip("JARVIS - Wake word listening")
            else:
                self.tray.setToolTip("JARVIS - Wake word starting")
        else:
            self.tray.wake_enabled = False
            self.tray.setToolTip("JARVIS - Active")

        self._trace(
            "wakeword_availability_changed",
            available=bool(available),
            wake_enabled=wake_enabled,
            message=message,
        )

        if message and wake_enabled:
            self._append_log(message)

    def _on_init_progress(self, label: str, progress: int) -> None:
        self._trace("pipeline_init_progress", label=label, progress=int(progress))
        self._append_log(f"Loading: {label} ({progress}%)")

    def _on_state_changed(self, state: str) -> None:
        self._trace("state_changed", state=state)
        self.sidebar.set_state(state)
        self.orb.set_state(state)
        self.waveform.set_state(state)

    def _on_new_message(self, role: str, text: str) -> None:
        self._trace("message_rendered", role=role, text_chars=len(str(text or "")))
        self.chat.add_message(role, text)

    def _on_intent(self, intent: str, confidence: float) -> None:
        self._trace("intent_classified", intent=intent, confidence=float(confidence))
        self.sidebar.add_intent(intent, confidence)

    def _on_intent_diagnostics(self, diagnostics: dict) -> None:
        try:
            self.sidebar.set_intent_diagnostics(diagnostics)
        except Exception:
            pass

    def _refresh_llm_status(self) -> None:
        if self.pipeline is None:
            self.llm_status_label.setText("LLM: waiting for pipeline")
            self._last_persistent_llm_status = "LLM: waiting for pipeline"
            return

        status = self.pipeline.get_llm_status()
        state = str(status.get("state", "initializing") or "initializing").lower()
        mode = str(status.get("mode", "-") or "-")
        message = str(status.get("message", "") or "")
        suffix = f" [{mode}]" if mode and mode != "-" else ""

        # Keep orb-adjacent status stable and only surface high-signal runtime info.
        if state in {"error", "unavailable"}:
            text = f"LLM: {state}{suffix}"
            if message:
                text = f"{text} - {message}"
            self._last_persistent_llm_status = text
            self.llm_status_label.setStyleSheet("color: rgba(255, 170, 150, 230);")
            self.llm_status_label.setText(text)
            return

        if state == "initializing":
            text = f"LLM: initializing{suffix}"
            if text != self._last_persistent_llm_status:
                self._last_persistent_llm_status = text
                self.llm_status_label.setStyleSheet("color: rgba(180, 220, 255, 220);")
                self.llm_status_label.setText(text)
            return

        if state == "ready":
            text = f"LLM: ready{suffix}"
            if text != self._last_persistent_llm_status:
                self._last_persistent_llm_status = text
                self.llm_status_label.setStyleSheet("color: rgba(180, 220, 255, 220);")
                self.llm_status_label.setText(text)
            return

        # Ignore transient states such as "processing" so random messages do not flicker below the orb.
        if self.llm_status_label.text() != self._last_persistent_llm_status:
            self.llm_status_label.setStyleSheet("color: rgba(180, 220, 255, 220);")
            self.llm_status_label.setText(self._last_persistent_llm_status)

    def _on_intent_model_changed(self, model_name: str) -> None:
        if self.pipeline is not None:
            self.pipeline.set_intent_model(model_name)
        self._trace("intent_model_changed", model_name=model_name)
        self._append_log(f"Intent model switched to {model_name}.")

    def _append_log(self, message: str) -> None:
        self._trace("log_line", message=message)
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self._logs.append(line)
        print(line, flush=True)
        if len(self._logs) > 500:
            self._logs = self._logs[-500:]

    def _on_verified_web_toggled(self, enabled: bool) -> None:
        if self.pipeline is None:
            return
        self.pipeline.set_realtime_web_enabled(bool(enabled))
        self._trace("verified_web_toggled", enabled=bool(enabled))
        self._append_log(f"Verified web mode {'enabled' if enabled else 'disabled'}.")

    def _analyze_image_from_file(self) -> None:
        if not self.pipeline:
            return

        self._trace("attach_image_dialog_opened")

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.gif)",
        )
        if not path:
            self._trace("attach_image_dialog_cancelled")
            return

        self._trace("attach_image_dialog_selected", path=path)

        self._attach_image(path, source="drive")

    def _attach_screenshot(self) -> None:
        self._trace("attach_screenshot_requested")
        self.showMinimized()
        QTimer.singleShot(300, self._do_attach_screenshot)

    def _do_attach_screenshot(self) -> None:
        try:
            _, path = self._screen_capture.capture_full(save=True)
        except Exception as exc:
            path = None
            trace_exception("ui.main_window", exc, event="attach_screenshot_failed")
            self._append_log(f"Screenshot capture failed: {exc}")
        finally:
            self.showNormal()
            self.activateWindow()

        if path is None:
            self._trace("attach_screenshot_failed")
            self._append_log("Screenshot capture failed.")
            return

        self._trace("attach_screenshot_completed", path=str(path))

        self._attach_image(str(path), source="screenshot")

    def _attach_camera_snapshot(self) -> None:
        self._trace("attach_camera_requested")
        ok, path, message = self._webcam_capture.capture_frame()
        if not ok or path is None:
            self._trace("attach_camera_failed", message=message)
            self._append_log(f"Camera capture failed: {message}")
            return
        self._trace("attach_camera_completed", path=str(path))
        self._attach_image(str(path), source="camera")

    def _attach_image(self, path: str, source: str = "drive") -> None:
        self._trace("attach_image_attempt", path=path, source=source)
        image_path = Path(path)
        if not image_path.exists():
            self._trace("attach_image_missing", path=path, source=source)
            self._append_log(f"Attached image does not exist: {path}")
            return

        self._attached_image_path = image_path
        self._attached_image_source = source
        self._last_image_path = image_path
        self._last_image_source = source

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.attachment_thumb.clear()
            self.inline_attachment_icon.clear()
            self.inline_attachment_icon.setVisible(False)
        else:
            scaled = pixmap.scaled(54, 54, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.attachment_thumb.setPixmap(scaled)
            inline_scaled = pixmap.scaled(
                20,
                20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.inline_attachment_icon.setPixmap(inline_scaled)
            self.inline_attachment_icon.setVisible(True)

        source_text = {
            "drive": "drive image",
            "camera": "camera snapshot",
            "screenshot": "screenshot",
        }.get(source, "image")

        self.attachment_label.setText(
            f"Attached {source_text}: {image_path.name}\nType your question and press Send."
        )
        self.attachment_bar.setVisible(True)
        self._refresh_attachment_actions()
        self._trace("attach_image_ready", path=str(image_path), source=source)
        self._append_log(f"Image attached ({source_text}): {image_path}")

    def _clear_attached_image(self, keep_last_preview: bool = True) -> None:
        self._trace(
            "clear_attached_image",
            keep_last_preview=bool(keep_last_preview),
            had_image=self._attached_image_path is not None,
        )
        current_path = self._attached_image_path
        current_source = self._attached_image_source
        self._attached_image_path = None
        self._attached_image_source = ""
        if not keep_last_preview:
            self._last_image_path = None
            self._last_image_source = ""
        elif current_path is not None:
            self._last_image_path = current_path
            self._last_image_source = current_source
        self.attachment_thumb.clear()
        self.inline_attachment_icon.clear()
        self.inline_attachment_icon.setVisible(False)
        self.attachment_label.setText("No image attached")
        self.attachment_bar.setVisible(False)
        self._refresh_attachment_actions()

    def _remove_attached_image(self) -> None:
        if self._attached_image_path is None:
            self._trace("remove_attached_image_skipped")
            self._append_log("No image is currently attached.")
            return
        self._clear_attached_image(keep_last_preview=True)
        self._trace("remove_attached_image_done")
        self._append_log("Attached image removed.")

    def _on_attachment_thumb_clicked(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self._view_last_image()

    def _view_last_image(self) -> None:
        image_path = self._attached_image_path
        if image_path is None or not image_path.exists():
            self._trace("preview_image_skipped_no_image")
            self._append_log("No image available to preview.")
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._trace("preview_image_failed_invalid_pixmap", path=str(image_path))
            self._append_log("Image preview failed: unsupported image format.")
            return

        self._trace("preview_image_opened", path=str(image_path))

        dialog = QDialog(self)
        dialog.setWindowTitle("Image Preview")
        dialog.resize(920, 620)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        source = self._attached_image_source or self._last_image_source or "image"
        source_text = {
            "drive": "uploaded image",
            "camera": "camera snapshot",
            "screenshot": "screenshot",
        }.get(source, source)

        hint = QLabel(f"Previewing the latest {source_text}: {image_path.name}", dialog)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        image_label = QLabel(dialog)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setPixmap(pixmap)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setWidget(image_label)
        layout.addWidget(scroll, 1)

        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

        dialog.exec()
        self._trace("preview_image_closed", path=str(image_path))

    def _refresh_attachment_actions(self) -> None:
        has_attached = self._attached_image_path is not None
        has_preview = has_attached and self._attached_image_path is not None and self._attached_image_path.exists()

        attached_source = self._attached_image_source
        self.analyze_file_action.setText("Attach Image [selected]" if has_attached and attached_source == "drive" else "Attach Image")
        self.attach_screenshot_action.setText(
            "Attach Screenshot [selected]" if has_attached and attached_source == "screenshot" else "Attach Screenshot"
        )
        self.attach_camera_action.setText(
            "Attach Camera Snapshot [selected]" if has_attached and attached_source == "camera" else "Attach Camera Snapshot"
        )

        self.view_attached_action.setEnabled(has_preview)
        self.remove_attached_action.setEnabled(has_attached)
        self.attachment_remove_btn.setEnabled(has_attached)

    def _set_send_button_mode(self, stop_mode: bool) -> None:
        stop_mode = bool(stop_mode)
        if self._send_button_stop_mode == stop_mode:
            return

        self._send_button_stop_mode = stop_mode
        self._trace("send_button_mode_changed", stop_mode=stop_mode)
        if stop_mode:
            self.send_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.send_btn.setToolTip("Stop")
        else:
            self.send_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
            self.send_btn.setToolTip("Send")

    def _on_send_or_stop_clicked(self) -> None:
        self._trace("send_or_stop_clicked", stop_mode=bool(self._send_button_stop_mode))
        if self._send_button_stop_mode:
            self._request_stop_response()
            return
        self._submit_text()

    def _request_stop_response(self) -> None:
        if self.pipeline is None:
            self._trace("stop_request_skipped", reason="pipeline_none")
            return
        if not (self._text_task_running or self._voice_task_running):
            self._trace("stop_request_skipped", reason="no_active_task")
            return

        try:
            self.pipeline.cancel_current_action()
        except Exception as exc:
            trace_exception("ui.main_window", exc, event="stop_request_failed")
            self._append_log(f"Stop request failed: {exc}")
            return

        self._trace("stop_request_sent")
        self._append_log("Stop requested.")
        self._stop_long_task_hint_timer()
        self.send_btn.setEnabled(False)
        self._set_send_button_mode(False)

    def _analyze_camera_snapshot(self) -> None:
        if not self.pipeline:
            return

        self._trace("camera_analyze_requested")
        self.chat.add_message("user", "Analyze camera snapshot")

        def worker() -> None:
            started = time.perf_counter()
            try:
                self.pipeline.analyze_camera()
                self._trace(
                    "camera_analyze_completed",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            except Exception as exc:
                trace_exception("ui.main_window", exc, event="camera_analyze_failed")
                if self.pipeline is not None:
                    self.pipeline.new_message.emit("assistant", f"Camera analysis error: {exc}")
                self.log_requested.emit(f"Camera analysis failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _submit_text(self) -> None:
        text = self.input_box.text().strip()
        if self.pipeline is None:
            self._trace("submit_text_skipped", reason="pipeline_none")
            return
        if self._text_task_running or self._voice_task_running or self._voice_capture_active:
            self._trace("submit_text_skipped", reason="task_or_capture_busy")
            return

        attached_path = self._attached_image_path
        if not text and attached_path is None:
            self._trace("submit_text_skipped", reason="empty_input")
            return

        self._trace(
            "submit_text_accepted",
            text_chars=len(text),
            has_attached_image=attached_path is not None,
        )

        self.input_box.clear()
        self._text_task_running = True
        self._set_send_button_mode(True)
        self.send_btn.setEnabled(True)
        self.mic_btn.setEnabled(False)
        self._on_state_changed("PROCESSING")
        self._configure_processing_hint(text, has_image=attached_path is not None, from_voice=False)

        if attached_path is not None:
            prompt = text or "Describe this image in detail."
            self.chat.add_message("user", prompt)
            self._clear_attached_image(keep_last_preview=True)

            def worker() -> None:
                started = time.perf_counter()
                try:
                    self.pipeline.analyze_image_file(str(attached_path), prompt=prompt)
                    self._trace(
                        "submit_text_image_completed",
                        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    )
                except Exception as exc:
                    trace_exception("ui.main_window", exc, event="submit_text_image_failed")
                    if self.pipeline is not None:
                        self.pipeline.new_message.emit("assistant", f"Image analysis error: {exc}")
                    self.log_requested.emit(f"Image analysis failed: {exc}")
                finally:
                    self.text_task_finished.emit()

            threading.Thread(target=worker, daemon=True).start()
            return

        def worker() -> None:
            started = time.perf_counter()
            try:
                self.pipeline.process_text(text)
                self._trace(
                    "submit_text_completed",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            except Exception as exc:
                trace_exception("ui.main_window", exc, event="submit_text_failed")
                if self.pipeline is not None:
                    self.pipeline.new_message.emit("assistant", f"Error: {exc}")
                self.log_requested.emit(f"Text processing error: {exc}")
            finally:
                self.text_task_finished.emit()

        threading.Thread(target=worker, daemon=True).start()

    def _on_text_task_complete(self) -> None:
        self._trace("text_task_complete")
        self._text_task_running = False
        self._stop_long_task_hint_timer()
        if self.pipeline is not None:
            self._set_send_button_mode(False)
            self.send_btn.setEnabled(True)
            if not self._voice_task_running and not self._voice_capture_active:
                self.mic_btn.setEnabled(True)

    def _set_mic_button_recording(self, recording: bool) -> None:
        if recording:
            self.mic_btn.setStyleSheet("background: rgba(255, 96, 96, 65);")
        else:
            self.mic_btn.setStyleSheet("")

    def _start_hold_to_talk(self) -> None:
        if not self.pipeline:
            self._trace("hold_to_talk_skipped", reason="pipeline_none")
            return
        if self._voice_task_running or self._voice_capture_active or self._text_task_running:
            self._trace("hold_to_talk_skipped", reason="task_or_capture_busy")
            return

        self._resume_wake_after_hold = False
        try:
            wake_cfg = self.pipeline.get_wakeword_settings()
            if bool(wake_cfg.get("enabled", False)) and self.pipeline.is_wakeword_available():
                self.pipeline.stop_wake_word()
                self._resume_wake_after_hold = True
        except Exception:
            self._resume_wake_after_hold = False

        self._trace(
            "hold_to_talk_started",
            resume_wake_after_hold=bool(self._resume_wake_after_hold),
        )

        self._voice_capture_active = True
        self._voice_capture_started_at = time.perf_counter()
        self._voice_capture_chunks = []
        self._voice_capture_stop.clear()
        self.send_btn.setEnabled(False)
        self._set_mic_button_recording(True)
        self._on_state_changed("LISTENING")
        self._append_log("Hold-to-talk started.")

        threading.Thread(target=self._capture_hold_to_talk_worker, daemon=True).start()

    def _stop_hold_to_talk(self) -> None:
        if not self._voice_capture_active:
            self._trace("hold_to_talk_stop_skipped", reason="capture_inactive")
            return
        self._trace("hold_to_talk_stop_requested")
        self._voice_capture_stop.set()

    def _capture_hold_to_talk_worker(self) -> None:
        self._trace("hold_capture_worker_started")
        local_chunks: list[np.ndarray] = []

        def callback(indata, _frames, _time_info, status) -> None:
            if status:
                self.log_requested.emit(f"Mic stream status: {status}")
            local_chunks.append(np.asarray(indata, dtype=np.float32).flatten())

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self._voice_sample_rate,
                dtype="float32",
                blocksize=1600,
                callback=callback,
            ):
                while not self._voice_capture_stop.wait(0.05):
                    if (time.perf_counter() - self._voice_capture_started_at) > 20.0:
                        self._voice_capture_stop.set()
                        break
        except Exception as exc:
            self._voice_capture_chunks = []
            trace_exception("ui.main_window", exc, event="hold_capture_worker_failed")
            self.log_requested.emit(f"Hold-to-talk capture error: {exc}")
            self.voice_capture_failed.emit(str(exc))
            return

        self._voice_capture_chunks = local_chunks
        self._trace("hold_capture_worker_completed", chunks=len(local_chunks))
        self.voice_capture_finished.emit()

    def _on_hold_capture_failed(self, error_text: str) -> None:
        self._trace("hold_capture_failed", error=error_text)
        self._voice_capture_active = False
        self._voice_capture_stop.clear()
        self._set_mic_button_recording(False)
        self._set_send_button_mode(False)
        self._on_state_changed("IDLE")
        self._stop_long_task_hint_timer()
        self._resume_wake_listener_if_needed()
        if self.pipeline is not None:
            self.send_btn.setEnabled(not self._text_task_running)
            self.mic_btn.setEnabled(True)
            self.pipeline.new_message.emit("assistant", f"Microphone error: {error_text}")

    def _on_hold_capture_finished(self) -> None:
        self._voice_capture_active = False
        self._voice_capture_stop.clear()
        self._set_mic_button_recording(False)

        if not self.pipeline:
            return

        chunks = self._voice_capture_chunks
        self._voice_capture_chunks = []
        if not chunks:
            self._trace("hold_capture_finished_no_audio")
            self._set_send_button_mode(False)
            self._on_state_changed("IDLE")
            self.send_btn.setEnabled(True)
            self.mic_btn.setEnabled(True)
            self._resume_wake_listener_if_needed()
            return

        audio = np.concatenate(chunks, axis=0)
        self._trace("hold_capture_audio_ready", samples=int(audio.size), sample_rate=self._voice_sample_rate)
        if audio.size < int(self._voice_sample_rate * 0.2):
            self._trace("hold_capture_audio_too_short", samples=int(audio.size))
            self._set_send_button_mode(False)
            self._on_state_changed("IDLE")
            self.send_btn.setEnabled(True)
            self.mic_btn.setEnabled(True)
            self.pipeline.new_message.emit("assistant", "Hold the mic button a bit longer so I can hear your full query.")
            self._resume_wake_listener_if_needed()
            return

        self._voice_task_running = True
        self._set_send_button_mode(True)
        self.send_btn.setEnabled(True)
        self.mic_btn.setEnabled(False)
        self._on_state_changed("PROCESSING")
        self._configure_processing_hint("", has_image=False, from_voice=True)

        def worker() -> None:
            started = time.perf_counter()
            try:
                self.pipeline.process_recorded_audio(audio, sample_rate=self._voice_sample_rate)
                self._trace(
                    "voice_task_completed",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            except Exception as exc:
                trace_exception("ui.main_window", exc, event="voice_task_failed")
                if self.pipeline is not None:
                    self.pipeline.new_message.emit("assistant", f"Voice error: {exc}")
                self.log_requested.emit(f"Voice processing error: {exc}")
            finally:
                self.voice_task_finished.emit()

        threading.Thread(target=worker, daemon=True).start()

    def _on_voice_task_complete(self) -> None:
        self._trace("voice_task_finish_signal")
        self._voice_task_running = False
        self._stop_long_task_hint_timer()
        if self.pipeline is not None:
            self._set_send_button_mode(False)
            self.mic_btn.setEnabled(True)
            if not self._text_task_running:
                self.send_btn.setEnabled(True)
        self._resume_wake_listener_if_needed()

    def _resume_wake_listener_if_needed(self) -> None:
        if not self._resume_wake_after_hold:
            self._trace("resume_wake_skipped", reason="flag_not_set")
            return
        self._resume_wake_after_hold = False
        if self.pipeline is None:
            self._trace("resume_wake_skipped", reason="pipeline_none")
            return
        try:
            wake_cfg = self.pipeline.get_wakeword_settings()
            if bool(wake_cfg.get("enabled", False)) and self.pipeline.is_wakeword_available():
                self.pipeline.start_wake_word()
                self._trace("resume_wake_started")
        except Exception:
            self._trace("resume_wake_failed")

    def _open_logs(self) -> None:
        self._trace("open_logs_dialog", log_lines=len(self._logs))
        dialog = QDialog(self)
        dialog.setWindowTitle("JARVIS Logs")
        dialog.setModal(True)
        dialog.resize(560, 420)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        viewer = QPlainTextEdit(dialog)
        viewer.setReadOnly(True)
        viewer.setFont(FONT_MONO)
        viewer.setPlainText("\n".join(self._logs) if self._logs else "No logs yet.")
        layout.addWidget(viewer, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        clear_btn = buttons.addButton("Clear Logs", QDialogButtonBox.ButtonRole.ActionRole)

        def clear_logs() -> None:
            self._logs.clear()
            viewer.setPlainText("No logs yet.")
            self._trace("logs_cleared")

        clear_btn.clicked.connect(clear_logs)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()
        self._trace("close_logs_dialog")

    def _open_settings(self) -> None:
        if not self.pipeline:
            self._trace("open_settings_skipped", reason="pipeline_not_ready")
            QMessageBox.information(self, "JARVIS Settings", "Pipeline is still starting. Please wait a moment.")
            return

        self._trace("open_settings_dialog")

        dialog = QDialog(self)
        dialog.setWindowTitle("JARVIS Settings")
        dialog.setModal(True)
        dialog.resize(560, 460)

        outer_layout = QVBoxLayout(dialog)
        outer_layout.setContentsMargins(16, 16, 16, 16)
        outer_layout.setSpacing(10)

        scroll_area = QScrollArea(dialog)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll_area, 1)

        settings_content = QWidget(scroll_area)
        scroll_area.setWidget(settings_content)

        layout = QVBoxLayout(settings_content)
        layout.setContentsMargins(4, 4, 8, 4)
        layout.setSpacing(10)

        wake_cfg = self.pipeline.get_wakeword_settings()
        wake_checkbox = QCheckBox("Enable passive wake-word listener", dialog)
        wake_checkbox.setChecked(bool(wake_cfg.get("enabled", False)))

        tts_cfg = self.pipeline.get_tts_settings()
        tts_checkbox = QCheckBox("Speak responses aloud (TTS)", dialog)
        tts_checkbox.setChecked(bool(tts_cfg.get("enabled", True)))

        tts_profile_combo = QComboBox(dialog)
        tts_profile_combo.addItem("FRIDAY", "female")
        tts_profile_combo.addItem("JARVIS", "male")
        tts_profile_idx = tts_profile_combo.findData(str(tts_cfg.get("profile", "female")))
        if tts_profile_idx >= 0:
            tts_profile_combo.setCurrentIndex(tts_profile_idx)

        profile_description = QLabel(dialog)
        profile_description.setWordWrap(True)
        profile_description.setFont(FONT_MONO)

        def refresh_profile_description() -> None:
            selected = str(tts_profile_combo.currentData() or "female")
            if selected == "female":
                profile_description.setText("FRIDAY: Female Replacement Intelligent Digital Assistant Youth")
            else:
                profile_description.setText("JARVIS: Just A Really Very Intelligent System")

        refresh_profile_description()
        tts_profile_combo.currentIndexChanged.connect(lambda _=0: refresh_profile_description())

        response_cfg = self.pipeline.get_response_settings()
        response_mode_combo = QComboBox(dialog)
        response_mode_combo.addItem("Brief (fast)", "brief")
        response_mode_combo.addItem("Normal (balanced)", "normal")
        response_mode_combo.addItem("Detailed (longer)", "detailed")
        response_mode_idx = response_mode_combo.findData(str(response_cfg.get("verbosity", "normal")))
        if response_mode_idx >= 0:
            response_mode_combo.setCurrentIndex(response_mode_idx)

        verified_web_checkbox = QCheckBox("Enable verified real-time web mode", dialog)
        verified_web_checkbox.setChecked(self.pipeline.is_realtime_web_enabled())

        compute_cfg = self.pipeline.get_compute_settings()
        compute_mode_combo = QComboBox(dialog)
        compute_mode_combo.addItem("Let JARVIS decide (dynamic)", "auto")
        compute_mode_combo.addItem("CPU only", "cpu")
        compute_mode_combo.addItem("GPU preferred", "gpu")

        current_compute_mode = str(compute_cfg.get("mode", "auto"))
        idx = compute_mode_combo.findData(current_compute_mode)
        if idx >= 0:
            compute_mode_combo.setCurrentIndex(idx)

        compute_caps = dict(compute_cfg.get("capabilities") or {})
        if compute_caps:
            gpu_supported = bool(compute_caps.get("gpu_supported", False))
            llm_gpu_supported = bool(compute_caps.get("llm_gpu_offload_supported", False))
            stt_cuda_ready = bool(compute_caps.get("stt_cuda_ready", False))
            stt_cuda_reason = str(compute_caps.get("stt_cuda_reason") or "")
            gpu_name = str(compute_caps.get("gpu_name") or "Not detected")
            providers = compute_caps.get("onnx_providers") or []
            providers_text = ", ".join(str(item) for item in providers) if providers else "None detected"
            stt_status_text = "Yes" if stt_cuda_ready else "No (Mic STT will use CPU)"
            stt_reason_line = f"\nSTT CUDA reason: {stt_cuda_reason}" if (stt_cuda_reason and not stt_cuda_ready) else ""
            compute_text = (
                f"Detected GPU: {gpu_name}\n"
                f"ONNX providers: {providers_text}\n"
                f"General GPU acceleration available: {'Yes' if gpu_supported else 'No (CPU fallback will be used)'}\n"
                f"LLM GPU offload available: {'Yes' if llm_gpu_supported else 'No (LLM will run on CPU)'}\n"
                f"STT CUDA runtime ready: {stt_status_text}"
                f"{stt_reason_line}"
            )
        else:
            compute_text = (
                "Compute capability scan is running in the background.\n"
                "Reopen settings in a moment to see full GPU/STT capability details."
            )

        compute_status = QLabel(compute_text, dialog)
        compute_status.setWordWrap(True)
        compute_status.setFont(FONT_MONO)

        wake_sensitivity_input = QDoubleSpinBox(dialog)
        wake_sensitivity_input.setRange(0.10, 0.95)
        wake_sensitivity_input.setSingleStep(0.01)
        wake_sensitivity_input.setDecimals(2)
        wake_sensitivity_input.setValue(float(wake_cfg.get("sensitivity", 0.35)))

        wake_phrases_list = QListWidget(dialog)
        wake_phrases_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        wake_phrases_list.setMinimumHeight(110)
        for phrase in wake_cfg.get("activation_phrases", []):
            phrase_text = str(phrase).strip().lower()
            if phrase_text:
                wake_phrases_list.addItem(phrase_text)

        wake_phrase_input = QLineEdit(dialog)
        wake_phrase_input.setPlaceholderText("Type a wake phrase and click Add")

        wake_phrase_add_btn = QPushButton("Add Phrase", dialog)
        wake_phrase_remove_btn = QPushButton("Remove Selected", dialog)

        strict_phrase_checkbox = QCheckBox("Require phrase prefix in transcript", dialog)
        strict_phrase_checkbox.setChecked(bool(wake_cfg.get("strict_phrase_prefix", False)))

        auto_restart_checkbox = QCheckBox("Auto-resume wake listener after assistant response", dialog)
        auto_restart_checkbox.setChecked(bool(wake_cfg.get("auto_restart_after_response", True)))

        follow_up_checkbox = QCheckBox("Listen for follow-up query after each wake response", dialog)
        follow_up_checkbox.setChecked(bool(wake_cfg.get("follow_up_after_response", True)))

        follow_up_timeout_input = QSpinBox(dialog)
        follow_up_timeout_input.setRange(3, 20)
        follow_up_timeout_input.setValue(int(wake_cfg.get("follow_up_timeout", 8)))

        follow_up_turns_input = QSpinBox(dialog)
        follow_up_turns_input.setRange(0, 3)
        follow_up_turns_input.setValue(int(wake_cfg.get("max_followup_turns", 1)))

        wake_restore_btn = QPushButton("Restore Wakeword Defaults", dialog)

        def add_wake_phrase() -> None:
            phrase_text = str(wake_phrase_input.text() or "").strip().lower()
            if not phrase_text:
                return
            existing = {wake_phrases_list.item(i).text().strip().lower() for i in range(wake_phrases_list.count())}
            if phrase_text in existing:
                wake_phrase_input.clear()
                return
            wake_phrases_list.addItem(phrase_text)
            wake_phrase_input.clear()

        def remove_selected_wake_phrases() -> None:
            for item in wake_phrases_list.selectedItems():
                wake_phrases_list.takeItem(wake_phrases_list.row(item))

        def restore_wake_defaults() -> None:
            defaults = dict(DEFAULT_WAKEWORD_CONFIG)
            wake_sensitivity_input.setValue(float(defaults.get("sensitivity", 0.35)))
            wake_phrases_list.clear()
            for phrase in defaults.get("activation_phrases", ["jarvis", "hey jarvis"]):
                phrase_text = str(phrase).strip().lower()
                if phrase_text:
                    wake_phrases_list.addItem(phrase_text)
            strict_phrase_checkbox.setChecked(bool(defaults.get("strict_phrase_prefix", False)))
            auto_restart_checkbox.setChecked(bool(defaults.get("auto_restart_after_response", True)))
            follow_up_checkbox.setChecked(bool(defaults.get("follow_up_after_response", True)))
            follow_up_timeout_input.setValue(int(defaults.get("follow_up_timeout", 8)))
            follow_up_turns_input.setValue(int(defaults.get("max_followup_turns", 1)))

        wake_phrase_add_btn.clicked.connect(add_wake_phrase)
        wake_phrase_input.returnPressed.connect(add_wake_phrase)
        wake_phrase_remove_btn.clicked.connect(remove_selected_wake_phrases)
        wake_restore_btn.clicked.connect(restore_wake_defaults)

        wake_phrase_add_row = QHBoxLayout()
        wake_phrase_add_row.setSpacing(8)
        wake_phrase_add_row.addWidget(wake_phrase_input, 1)
        wake_phrase_add_row.addWidget(wake_phrase_add_btn)

        wake_phrase_actions_row = QHBoxLayout()
        wake_phrase_actions_row.setSpacing(8)
        wake_phrase_actions_row.addWidget(wake_phrase_remove_btn)
        wake_phrase_actions_row.addWidget(wake_restore_btn)

        wake_config_path = QLabel(f"Wakeword config path: {WAKEWORD_CONFIG_PATH}", dialog)
        wake_config_path.setWordWrap(True)
        wake_config_path.setFont(FONT_MONO)

        llm_ready = bool(
            self.pipeline.router
            and self.pipeline.router.llm is not None
            and self.pipeline.router.llm.is_ready()
        )
        stt_status = self.pipeline.get_stt_status()
        stt_available = bool(stt_status.get("available", False))
        stt_initialized = bool(stt_status.get("initialized", False))
        stt_reason = str(stt_status.get("reason", "") or "")

        wake_status = self.pipeline.get_wakeword_status()
        wake_enabled = bool(wake_status.get("enabled", False))
        wake_ready = bool(wake_status.get("available", False))
        wake_initializing = bool(wake_status.get("initializing", False))
        wake_reason = str(wake_status.get("reason", "") or "")

        if stt_available and stt_initialized:
            stt_text = "Ready"
        elif stt_available:
            stt_text = "Ready (lazy init)"
        else:
            stt_text = "Unavailable"

        if wake_enabled and wake_ready:
            wake_text = "Available"
        elif wake_enabled and wake_initializing:
            wake_text = "Starting in background"
        elif wake_enabled:
            wake_text = "Unavailable"
        else:
            wake_text = "Disabled"

        status = QLabel(
            f"Local LLM: {'Ready' if llm_ready else 'Will initialize on demand'}\n"
            f"Microphone STT: {stt_text}\n"
            f"Wake word: {wake_text}"
            f"{(f'\nSTT note: {stt_reason}' if (stt_reason and not stt_initialized) else '')}",
            dialog,
        )
        if wake_enabled and not wake_ready and wake_reason:
            status.setText(status.text() + f"\nWake note: {wake_reason}")
        status.setWordWrap(True)
        status.setFont(FONT_MONO)

        app_status = QLabel("App index: status available via Rescan Apps", dialog)
        app_status.setWordWrap(True)
        app_status.setFont(FONT_MONO)

        music_status = QLabel("Music index: status available via Rescan Music", dialog)
        music_status.setWordWrap(True)
        music_status.setFont(FONT_MONO)

        rescan_apps_btn = QPushButton("Rescan Apps", dialog)
        rescan_music_btn = QPushButton("Rescan Music", dialog)

        def rescan_apps() -> None:
            self._trace("settings_rescan_apps_requested")
            app_status.setText("App index: rescanning...")

            def worker() -> None:
                try:
                    from actions import app_control

                    result = app_control.rescan_app_index()
                    message = str(result.get("response_text", "App index scan completed."))
                except Exception as exc:
                    message = f"App index scan failed: {exc}"
                    trace_exception("ui.main_window", exc, event="settings_rescan_apps_failed")
                self._trace("settings_rescan_apps_completed", message=message)
                self.log_requested.emit(message)
                QTimer.singleShot(0, lambda: app_status.setText(f"App index: {message}"))

            threading.Thread(target=worker, daemon=True).start()

        def rescan_music() -> None:
            self._trace("settings_rescan_music_requested")
            music_status.setText("Music index: rescanning...")

            def worker() -> None:
                try:
                    from actions import media_control

                    result = media_control.rescan_media_index()
                    message = str(result.get("response_text", "Music index scan completed."))
                except Exception as exc:
                    message = f"Music index scan failed: {exc}"
                    trace_exception("ui.main_window", exc, event="settings_rescan_music_failed")
                self._trace("settings_rescan_music_completed", message=message)
                self.log_requested.emit(message)
                QTimer.singleShot(0, lambda: music_status.setText(f"Music index: {message}"))

            threading.Thread(target=worker, daemon=True).start()

        rescan_apps_btn.clicked.connect(rescan_apps)
        rescan_music_btn.clicked.connect(rescan_music)

        rescan_row = QHBoxLayout()
        rescan_row.setSpacing(8)
        rescan_row.addWidget(rescan_apps_btn)
        rescan_row.addWidget(rescan_music_btn)

        layout.addWidget(wake_checkbox)
        layout.addWidget(tts_checkbox)
        layout.addWidget(QLabel("Voice profile", dialog))
        layout.addWidget(tts_profile_combo)
        layout.addWidget(profile_description)
        layout.addWidget(QLabel("Response verbosity", dialog))
        layout.addWidget(response_mode_combo)
        layout.addWidget(verified_web_checkbox)
        layout.addWidget(QLabel("Compute backend mode", dialog))
        layout.addWidget(compute_mode_combo)
        layout.addWidget(compute_status)
        layout.addWidget(QLabel("Wakeword sensitivity", dialog))
        layout.addWidget(wake_sensitivity_input)
        layout.addWidget(QLabel("Wakeword phrases", dialog))
        layout.addWidget(wake_phrases_list)
        layout.addLayout(wake_phrase_add_row)
        layout.addLayout(wake_phrase_actions_row)
        layout.addWidget(strict_phrase_checkbox)
        layout.addWidget(auto_restart_checkbox)
        layout.addWidget(follow_up_checkbox)
        layout.addWidget(QLabel("Follow-up timeout (seconds)", dialog))
        layout.addWidget(follow_up_timeout_input)
        layout.addWidget(QLabel("Max follow-up turns", dialog))
        layout.addWidget(follow_up_turns_input)
        layout.addWidget(wake_config_path)
        layout.addWidget(status)
        layout.addWidget(app_status)
        layout.addWidget(music_status)
        layout.addLayout(rescan_row)
        layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer_layout.addWidget(buttons)

        self._trace(
            "settings_dialog_ready",
            wake_enabled=bool(wake_cfg.get("enabled", False)),
            tts_enabled=bool(tts_cfg.get("enabled", True)),
            tts_profile=str(tts_cfg.get("profile", "female")),
            response_verbosity=str(response_cfg.get("verbosity", "normal")),
            compute_mode=current_compute_mode,
            realtime_web_enabled=bool(self.pipeline.is_realtime_web_enabled()),
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._trace("settings_dialog_cancelled")
            return

        self.pipeline.set_tts_enabled(tts_checkbox.isChecked())
        selected_profile = str(tts_profile_combo.currentData() or "female")
        self.pipeline.set_tts_profile(selected_profile)
        selected_verbosity = str(response_mode_combo.currentData() or "normal")
        self.pipeline.set_response_verbosity(selected_verbosity)
        self.pipeline.set_realtime_web_enabled(verified_web_checkbox.isChecked())
        self._trace(
            "settings_apply_runtime",
            tts_enabled=bool(tts_checkbox.isChecked()),
            tts_profile=selected_profile,
            response_verbosity=selected_verbosity,
            realtime_web_enabled=bool(verified_web_checkbox.isChecked()),
        )
        self._append_log(f"TTS profile set to {selected_profile}.")
        self._append_log(f"Response verbosity set to {selected_verbosity}.")

        selected_compute_mode = str(compute_mode_combo.currentData() or "auto")
        updated_compute = self.pipeline.set_compute_mode(selected_compute_mode)
        active_compute_mode = str(updated_compute.get("mode", selected_compute_mode))
        active_caps = dict(updated_compute.get("capabilities") or {})
        gpu_active_available = bool(active_caps.get("gpu_supported", False))
        llm_gpu_active_available = bool(active_caps.get("llm_gpu_offload_supported", False))
        if active_caps:
            self._append_log(
                f"Compute mode set to {active_compute_mode.upper()}"
                f" ({'GPU acceleration available' if gpu_active_available else 'CPU fallback active'}; "
                f"{'LLM GPU enabled' if llm_gpu_active_available else 'LLM CPU fallback'})."
            )
        else:
            self._append_log(
                f"Compute mode set to {active_compute_mode.upper()} (capability refresh in progress)."
            )
        self._trace(
            "settings_apply_compute",
            selected_compute_mode=selected_compute_mode,
            active_compute_mode=active_compute_mode,
            gpu_supported=gpu_active_available,
            llm_gpu_supported=llm_gpu_active_available,
            has_capabilities=bool(active_caps),
        )

        if active_caps and active_compute_mode == "gpu" and not gpu_active_available:
            QMessageBox.information(
                self,
                "JARVIS Settings",
                "GPU mode was selected, but no compatible GPU runtime is currently available."
                " JARVIS will continue with CPU fallback until GPU dependencies are available.",
            )
        elif active_caps and active_compute_mode == "gpu" and not llm_gpu_active_available:
            QMessageBox.information(
                self,
                "JARVIS Settings",
                "GPU mode is enabled for supported components, but your local LLM runtime "
                "(llama-cpp-python) does not currently support GPU offload in this environment. "
                "LLM responses will run on CPU until GPU-enabled llama-cpp is installed.",
            )

        sensitivity = float(wake_sensitivity_input.value())

        phrases = [wake_phrases_list.item(i).text().strip().lower() for i in range(wake_phrases_list.count())]
        phrases = [item for item in phrases if item]
        if not phrases:
            phrases = list(DEFAULT_WAKEWORD_CONFIG.get("activation_phrases", ["jarvis", "hey jarvis"]))
        strict_phrase = strict_phrase_checkbox.isChecked()
        auto_restart_after_response = auto_restart_checkbox.isChecked()
        follow_up_after_response = follow_up_checkbox.isChecked()

        follow_up_timeout = int(follow_up_timeout_input.value())
        max_followup_turns = int(follow_up_turns_input.value())

        wake_enabled = wake_checkbox.isChecked()

        self._append_log("Wakeword settings saved through structured controls.")

        self.pipeline.update_wakeword_settings(
            enabled=wake_enabled,
            sensitivity=sensitivity,
            activation_phrases=phrases,
            strict_phrase_prefix=strict_phrase,
            auto_restart_after_response=auto_restart_after_response,
            follow_up_after_response=follow_up_after_response,
            follow_up_timeout=follow_up_timeout,
            max_followup_turns=max_followup_turns,
        )
        self._trace(
            "settings_apply_wakeword",
            enabled=bool(wake_enabled),
            sensitivity=float(sensitivity),
            activation_phrases=list(phrases),
            strict_phrase_prefix=bool(strict_phrase),
            auto_restart_after_response=bool(auto_restart_after_response),
            follow_up_after_response=bool(follow_up_after_response),
            follow_up_timeout=int(follow_up_timeout),
            max_followup_turns=int(max_followup_turns),
        )

        if wake_enabled:
            self.enable_wake_word()
        else:
            self.disable_wake_word()

        self.verified_web_action.blockSignals(True)
        self.verified_web_action.setChecked(self.pipeline.is_realtime_web_enabled())
        self.verified_web_action.blockSignals(False)
        self._trace("settings_apply_complete")

    def enable_wake_word(self) -> None:
        if not self.pipeline:
            self._trace("enable_wake_skipped", reason="pipeline_none")
            return
        self._trace("enable_wake_requested")
        self.pipeline.start_wake_word()
        wake_status = self.pipeline.get_wakeword_status()
        if bool(wake_status.get("initializing", False)):
            self.tray.wake_enabled = False
            self._trace("enable_wake_initializing")
            self._append_log("Wake-word listener is initializing in the background.")
            return
        if not bool(wake_status.get("available", False)):
            reason = str(wake_status.get("reason", "") or "")
            self._trace("enable_wake_unavailable", reason=reason)
            self.chat.add_message(
                "assistant",
                reason
                or "Wake-word backend is unavailable in this runtime. You can keep using the Mic button for explicit voice input.",
            )
            self.tray.wake_enabled = False
            return

        self.tray.wake_enabled = True
        self._trace("enable_wake_ready")

    def disable_wake_word(self) -> None:
        self._trace("disable_wake_requested")
        if self.pipeline:
            self.pipeline.stop_wake_word()
        self.tray.wake_enabled = False
        self._trace("disable_wake_complete")

    def _shutdown_for_exit(self) -> None:
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        self._trace("shutdown_for_exit_started")

        if self.pipeline is not None:
            try:
                self.pipeline.shutdown()
            except Exception as exc:
                trace_exception("ui.main_window", exc, event="shutdown_pipeline_failed")

        try:
            self.tray.hide()
        except Exception:
            pass

        self._trace("shutdown_for_exit_completed")

    def _prompt_close_behavior(self) -> str:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Exit JARVIS")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText("Choose what to do when closing JARVIS.")
        dialog.setInformativeText("Run in background, close completely, or cancel.")

        background_btn = dialog.addButton("Run in Background", QMessageBox.ButtonRole.AcceptRole)
        close_btn = dialog.addButton("Close Completely", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = dialog.addButton(QMessageBox.StandardButton.Cancel)

        dialog.setDefaultButton(background_btn)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked == close_btn:
            return "close"
        if clicked == background_btn:
            return "background"
        if clicked == cancel_btn:
            return "cancel"
        return "cancel"

    def force_quit(self) -> None:
        self._trace("force_quit_requested")
        self._shutdown_for_exit()
        self.close()

    def closeEvent(self, event) -> None:
        if self.tray.isVisible():
            choice = self._prompt_close_behavior()
            if choice == "cancel":
                self._trace("close_event_cancelled")
                event.ignore()
                return
            if choice == "background":
                self._trace("close_event_to_tray")
                event.ignore()
                self.hide()
                self.tray.showMessage("JARVIS", "JARVIS is still running in system tray.")
                return

            self._trace("close_event_exit_selected")
            self._shutdown_for_exit()
            event.accept()
            app = QApplication.instance()
            if app is not None:
                app.quit()
            return

        self._trace("close_event_accept")
        self._shutdown_for_exit()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos is not None:
            self._trace("window_drag_end")
        self._drag_pos = None




