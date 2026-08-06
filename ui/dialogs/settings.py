"""The settings window.

Replaces a ~350-line flat QVBoxLayout that lived inline in main_window.py --
every field a bare label above its control, in one column, no grouping, no
headings. It was the one surface the previous overhaul never reached.

Apply semantics are deliberately mixed, and the split is not an oversight:

  Appearance applies LIVE and reverts on Cancel. A theme picker that only
  takes effect after you press Save is useless, because the entire point is
  seeing the change. The previous values are snapshotted on open and put
  back if the dialog is rejected.

  Everything else stays BUFFERED until Save, preserving the existing
  contract exactly. Live-applying wake-word settings would tear down and
  re-initialise the detector on every keystroke.

The Save order matches what main_window did before, because several of these
setters have side effects on each other: tts_enabled, tts_profile (which also
drives branding and therefore the persona theme), verbosity, realtime web,
compute mode, wake word, then enable/disable the listener.
"""

from __future__ import annotations

import threading
from typing import Any

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.session_logging import trace_event, trace_exception
from speech.wakeword_config import DEFAULT_WAKEWORD_CONFIG, WAKEWORD_CONFIG_PATH
from ui.dialogs.rows import SettingsCard, SettingsPage
from ui.native import apply_dark_titlebar
from ui.theme import (
    body_font,
    manager as theme_manager,
    mono_font,
    theme,
)
from ui.theme.presets import PRESET_LABELS
from ui.ui_settings import ANIMATION_QUALITIES, load_ui_settings, update_ui_settings


PAGES = (
    ("General", "general"),
    ("Voice & speech", "voice"),
    ("Wake word", "wakeword"),
    ("Appearance", "appearance"),
    ("Compute", "compute"),
    ("Indexes", "indexes"),
    ("About", "about"),
)


class Banner(QFrame):
    """Inline notice. Replaces the QMessageBox interruptions."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("banner")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        self.label = QLabel("", self)
        self.label.setWordWrap(True)
        self.label.setFont(body_font(11))
        layout.addWidget(self.label, 1)
        self.setVisible(False)

    def show_message(self, text: str, tone: str = "info") -> None:
        self.label.setText(str(text or ""))
        self.setProperty("tone", tone)
        # A dynamic property only takes effect after the style is re-polished.
        self.style().unpolish(self)
        self.style().polish(self)
        self.setVisible(bool(text))


class SettingsWindow(QDialog):
    def __init__(self, host, parent=None) -> None:
        super().__init__(parent or host)
        self._host = host
        self._pipeline = host.pipeline
        self._name = getattr(host, "_assistant_name", "JARVIS")

        self.setWindowTitle(f"{self._name} Settings")
        self.setModal(True)
        self.resize(900, 640)
        self.setMinimumSize(760, 520)

        # Snapshot for the live-applying Appearance page, so Cancel can undo.
        self._theme_snapshot = (
            theme_manager().preset_key,
            theme_manager().accent_override,
            str(load_ui_settings().get("animation_quality", "auto")),
        )
        self._appearance_committed = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        self.rail = QListWidget(self)
        self.rail.setObjectName("settingsRail")
        self.rail.setFixedWidth(196)
        self.rail.setFrameShape(QFrame.Shape.NoFrame)
        self.rail.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for label, _key in PAGES:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, 38))
            self.rail.addItem(item)
        body.addWidget(self.rail)

        divider = QFrame(self)
        divider.setObjectName("dividerV")
        divider.setFixedWidth(1)
        body.addWidget(divider)

        self.stack = QStackedWidget(self)
        body.addWidget(self.stack, 1)

        self.banner = Banner(self)

        self._pages: dict[str, SettingsPage] = {}
        for label, key in PAGES:
            page = SettingsPage(label, self)
            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
            self._pages[key] = page

        self.rail.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.rail.setCurrentRow(0)

        footer = QWidget(self)
        footer.setObjectName("settingsFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 12, 20, 12)
        footer_layout.setSpacing(10)
        footer_layout.addWidget(self.banner, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            footer,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setObjectName("primary")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        footer_layout.addWidget(self.buttons, 0)
        root.addWidget(footer)

        self._build_general()
        self._build_voice()
        self._build_wakeword()
        self._build_appearance()
        self._build_compute()
        self._build_indexes()
        self._build_about()
        for page in self._pages.values():
            page.finish()

        self.accepted.connect(self._apply)
        self.rejected.connect(self._revert_appearance)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # winId() is only meaningful once the window is realised, and the
        # attribute has to be re-applied whenever the theme flips to light.
        apply_dark_titlebar(self, theme().palette.is_dark)

    # -- General ----------------------------------------------------------

    def _build_general(self) -> None:
        page = self._pages["general"]
        card = page.add_card()

        self.verified_web_check = QCheckBox(self)
        self.verified_web_check.setChecked(bool(self._pipeline.is_realtime_web_enabled()))
        card.add_row(
            "Verified web mode",
            self.verified_web_check,
            "Cross-checks answers against several live sources before replying. "
            "Slower, and it sends your query to the web.",
        )

    # -- Voice ------------------------------------------------------------

    def _build_voice(self) -> None:
        page = self._pages["voice"]
        tts_cfg = self._pipeline.get_tts_settings()
        response_cfg = self._pipeline.get_response_settings()
        card = page.add_card()

        self.tts_check = QCheckBox(self)
        self.tts_check.setChecked(bool(tts_cfg.get("enabled", True)))
        card.add_row("Speak responses aloud", self.tts_check,
                     "Read replies out with text to speech.")

        self.tts_profile_combo = QComboBox(self)
        self.tts_profile_combo.addItem("FRIDAY", "female")
        self.tts_profile_combo.addItem("JARVIS", "male")
        index = self.tts_profile_combo.findData(str(tts_cfg.get("profile", "female")))
        if index >= 0:
            self.tts_profile_combo.setCurrentIndex(index)
        self._profile_row = card.add_row("Voice profile", self.tts_profile_combo)
        self.tts_profile_combo.currentIndexChanged.connect(
            lambda _=0: self._refresh_profile_description())
        self._refresh_profile_description()

        self.verbosity_combo = QComboBox(self)
        self.verbosity_combo.addItem("Brief (fast)", "brief")
        self.verbosity_combo.addItem("Normal (balanced)", "normal")
        self.verbosity_combo.addItem("Detailed (longer)", "detailed")
        index = self.verbosity_combo.findData(str(response_cfg.get("verbosity", "normal")))
        if index >= 0:
            self.verbosity_combo.setCurrentIndex(index)
        card.add_row("Response length", self.verbosity_combo,
                     "How much detail replies carry.")

    def _refresh_profile_description(self) -> None:
        selected = str(self.tts_profile_combo.currentData() or "female")
        if selected == "female":
            text = "FRIDAY - Female Replacement Intelligent Digital Assistant Youth"
        else:
            text = "JARVIS - Just A Really Very Intelligent System"
        self._profile_row.set_description(
            text + ". Also sets the assistant's name and, on the automatic "
            "theme, its accent colour.")

    # -- Wake word --------------------------------------------------------

    def _build_wakeword(self) -> None:
        page = self._pages["wakeword"]
        cfg = self._pipeline.get_wakeword_settings()

        card = page.add_card()
        self.wake_check = QCheckBox(self)
        self.wake_check.setChecked(bool(cfg.get("enabled", False)))
        card.add_row("Passive listening", self.wake_check,
                     "Listen continuously for the wake phrase.")

        self.wake_sensitivity = QDoubleSpinBox(self)
        self.wake_sensitivity.setRange(0.10, 0.95)
        self.wake_sensitivity.setSingleStep(0.01)
        self.wake_sensitivity.setDecimals(2)
        self.wake_sensitivity.setValue(float(cfg.get("sensitivity", 0.35)))
        card.add_row("Sensitivity", self.wake_sensitivity,
                     "Lower triggers more easily and misfires more often.")

        self.strict_prefix_check = QCheckBox(self)
        self.strict_prefix_check.setChecked(bool(cfg.get("strict_phrase_prefix", False)))
        card.add_row("Require the phrase as a prefix", self.strict_prefix_check,
                     "Only act when the transcript starts with a wake phrase.")

        page.add_section("Phrases")
        phrase_card = page.add_card()

        self.phrase_list = QListWidget(self)
        self.phrase_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.phrase_list.setMinimumHeight(120)
        for phrase in cfg.get("activation_phrases", []):
            text = str(phrase).strip().lower()
            if text:
                self.phrase_list.addItem(text)
        phrase_card.add_row("Wake phrases", self.phrase_list, stacked=True)

        editor = QWidget(self)
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(8)

        add_row = QHBoxLayout()
        add_row.setSpacing(8)
        self.phrase_input = QLineEdit(self)
        self.phrase_input.setPlaceholderText("Type a wake phrase and press Add")
        add_btn = QPushButton("Add", self)
        add_btn.clicked.connect(self._add_phrase)
        self.phrase_input.returnPressed.connect(self._add_phrase)
        add_row.addWidget(self.phrase_input, 1)
        add_row.addWidget(add_btn)
        editor_layout.addLayout(add_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        remove_btn = QPushButton("Remove selected", self)
        remove_btn.clicked.connect(self._remove_phrases)
        restore_btn = QPushButton("Restore defaults", self)
        restore_btn.clicked.connect(self._restore_wake_defaults)
        action_row.addWidget(remove_btn)
        action_row.addWidget(restore_btn)
        action_row.addStretch(1)
        editor_layout.addLayout(action_row)
        phrase_card.add(editor)

        page.add_section("After a response")
        follow_card = page.add_card()

        self.auto_restart_check = QCheckBox(self)
        self.auto_restart_check.setChecked(bool(cfg.get("auto_restart_after_response", True)))
        follow_card.add_row("Resume listening", self.auto_restart_check,
                            "Go back to passive listening once a reply finishes.")

        self.follow_up_check = QCheckBox(self)
        self.follow_up_check.setChecked(bool(cfg.get("follow_up_after_response", True)))
        follow_card.add_row("Listen for a follow-up", self.follow_up_check,
                            "Stay open for one more question without the wake phrase.")

        self.follow_up_timeout = QSpinBox(self)
        self.follow_up_timeout.setRange(3, 20)
        self.follow_up_timeout.setSuffix(" s")
        self.follow_up_timeout.setValue(int(cfg.get("follow_up_timeout", 8)))
        follow_card.add_row("Follow-up timeout", self.follow_up_timeout)

        self.follow_up_turns = QSpinBox(self)
        self.follow_up_turns.setRange(0, 3)
        self.follow_up_turns.setValue(int(cfg.get("max_followup_turns", 1)))
        follow_card.add_row("Maximum follow-up turns", self.follow_up_turns)

        path_label = QLabel(f"Config file: {WAKEWORD_CONFIG_PATH}", self)
        path_label.setObjectName("readout")
        path_label.setWordWrap(True)
        path_label.setFont(mono_font(10))
        page.add_widget(path_label)

    def _add_phrase(self) -> None:
        text = str(self.phrase_input.text() or "").strip().lower()
        if not text:
            return
        existing = {self.phrase_list.item(i).text().strip().lower()
                    for i in range(self.phrase_list.count())}
        if text not in existing:
            self.phrase_list.addItem(text)
        self.phrase_input.clear()

    def _remove_phrases(self) -> None:
        for item in self.phrase_list.selectedItems():
            self.phrase_list.takeItem(self.phrase_list.row(item))

    def _restore_wake_defaults(self) -> None:
        defaults = dict(DEFAULT_WAKEWORD_CONFIG)
        self.wake_sensitivity.setValue(float(defaults.get("sensitivity", 0.35)))
        self.phrase_list.clear()
        for phrase in defaults.get("activation_phrases", ["jarvis", "hey jarvis"]):
            text = str(phrase).strip().lower()
            if text:
                self.phrase_list.addItem(text)
        self.strict_prefix_check.setChecked(bool(defaults.get("strict_phrase_prefix", False)))
        self.auto_restart_check.setChecked(bool(defaults.get("auto_restart_after_response", True)))
        self.follow_up_check.setChecked(bool(defaults.get("follow_up_after_response", True)))
        self.follow_up_timeout.setValue(int(defaults.get("follow_up_timeout", 8)))
        self.follow_up_turns.setValue(int(defaults.get("max_followup_turns", 1)))

    # -- Appearance -------------------------------------------------------

    def _build_appearance(self) -> None:
        page = self._pages["appearance"]
        ui_cfg = load_ui_settings()

        note = QLabel("Appearance changes apply immediately. Cancel puts them back.",
                      self)
        note.setObjectName("caption")
        note.setWordWrap(True)
        page.add_widget(note)

        card = page.add_card()

        self.theme_combo = QComboBox(self)
        self.theme_combo.addItem("Follow the assistant", "auto")
        for key, label in PRESET_LABELS.items():
            self.theme_combo.addItem(label, key)
        index = self.theme_combo.findData(theme_manager().preset_key)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_selected)
        card.add_row("Theme", self.theme_combo,
                     "Follow the assistant means JARVIS is gold and FRIDAY is blue.")

        accent_widget = QWidget(self)
        accent_layout = QHBoxLayout(accent_widget)
        accent_layout.setContentsMargins(0, 0, 0, 0)
        accent_layout.setSpacing(8)
        self.accent_swatch = QFrame(self)
        self.accent_swatch.setObjectName("accentSwatch")
        self.accent_swatch.setFixedSize(24, 24)
        accent_layout.addWidget(self.accent_swatch)
        pick_btn = QPushButton("Choose...", self)
        pick_btn.clicked.connect(self._pick_accent)
        accent_layout.addWidget(pick_btn)
        reset_btn = QPushButton("Reset", self)
        reset_btn.clicked.connect(self._reset_accent)
        accent_layout.addWidget(reset_btn)
        card.add_row("Accent colour", accent_widget,
                     "Overrides the theme's own accent.")
        self._refresh_accent_swatch()

        self.animation_combo = QComboBox(self)
        for value in ANIMATION_QUALITIES:
            self.animation_combo.addItem(value.title(), value)
        index = self.animation_combo.findData(str(ui_cfg.get("animation_quality", "auto")))
        if index >= 0:
            self.animation_combo.setCurrentIndex(index)
        self.animation_combo.currentIndexChanged.connect(self._on_animation_quality)
        card.add_row("Animation", self.animation_combo,
                     "Minimal replaces the animated mark with a static one.")

        page.add_section("Window")
        window_card = page.add_card()

        chrome = getattr(self._host, "_chrome", None)
        self.frameless_check = QCheckBox(self)
        self.frameless_check.setChecked(bool(ui_cfg.get("frameless", True)))
        window_card.add_row(
            "Custom window frame", self.frameless_check,
            "Draw our own title bar instead of the system one. "
            "Takes effect on restart.")

        acrylic_state = "on" if (chrome is not None and chrome.acrylic_active) else "off"
        rounded = bool(chrome is not None and chrome.supports_rounded_corners)
        detail = QLabel(
            f"Backdrop blur is currently {acrylic_state}. "
            + ("Rounded corners are supported on this version of Windows."
               if rounded else
               "Windows 10 cannot round the corners of a blurred window, so the "
               "window stays square here."),
            self)
        detail.setObjectName("caption")
        detail.setWordWrap(True)
        page.add_widget(detail)

    def _on_theme_selected(self) -> None:
        theme_manager().set_preset(str(self.theme_combo.currentData() or "auto"))
        self._refresh_accent_swatch()

    def _pick_accent(self) -> None:
        current = QColor(theme_manager().accent_override or theme().palette.accent)
        chosen = QColorDialog.getColor(current, self, "Accent colour")
        if chosen.isValid():
            theme_manager().set_accent(chosen)
            self._refresh_accent_swatch()

    def _reset_accent(self) -> None:
        theme_manager().set_accent("")
        self._refresh_accent_swatch()

    def _refresh_accent_swatch(self) -> None:
        colour = theme().palette.accent
        self.accent_swatch.setStyleSheet(
            f"background: {colour.name()}; border-radius: 6px;")

    def _on_animation_quality(self) -> None:
        value = str(self.animation_combo.currentData() or "auto")
        update_ui_settings(animation_quality=value)
        applier = getattr(self._host, "apply_animation_quality", None)
        if callable(applier):
            applier(value)

    def _revert_appearance(self) -> None:
        if self._appearance_committed:
            return
        preset, accent, animation = self._theme_snapshot
        theme_manager().set_preset(preset)
        theme_manager().set_accent(accent)
        update_ui_settings(animation_quality=animation)
        applier = getattr(self._host, "apply_animation_quality", None)
        if callable(applier):
            applier(animation)
        trace_event("ui.settings", "appearance_reverted", preset=preset)

    # -- Compute ----------------------------------------------------------

    def _build_compute(self) -> None:
        page = self._pages["compute"]
        cfg = self._pipeline.get_compute_settings()
        card = page.add_card()

        self.compute_combo = QComboBox(self)
        self.compute_combo.addItem(f"Let {self._name} decide", "auto")
        self.compute_combo.addItem("CPU only", "cpu")
        self.compute_combo.addItem("GPU preferred", "gpu")
        index = self.compute_combo.findData(str(cfg.get("mode", "auto")))
        if index >= 0:
            self.compute_combo.setCurrentIndex(index)
        card.add_row("Compute mode", self.compute_combo)

        caps = dict(cfg.get("capabilities") or {})
        readout = QLabel(self._capability_text(caps), self)
        readout.setObjectName("readout")
        readout.setWordWrap(True)
        readout.setFont(mono_font(10))
        page.add_widget(readout)

    @staticmethod
    def _capability_text(caps: dict[str, Any]) -> str:
        if not caps:
            return ("Capability scan is still running in the background.\n"
                    "Reopen settings shortly for the full GPU and STT report.")
        providers = caps.get("onnx_providers") or []
        providers_text = ", ".join(str(p) for p in providers) if providers else "none detected"
        stt_ready = bool(caps.get("stt_cuda_ready", False))
        stt_reason = str(caps.get("stt_cuda_reason") or "")
        lines = [
            f"GPU                 {caps.get('gpu_name') or 'not detected'}",
            f"ONNX providers      {providers_text}",
            f"GPU acceleration    {'available' if caps.get('gpu_supported') else 'unavailable, CPU fallback'}",
            f"LLM GPU offload     {'available' if caps.get('llm_gpu_offload_supported') else 'unavailable, LLM on CPU'}",
            f"STT CUDA runtime    {'ready' if stt_ready else 'not ready, STT on CPU'}",
        ]
        if stt_reason and not stt_ready:
            lines.append(f"STT note            {stt_reason}")
        return "\n".join(lines)

    # -- Indexes ----------------------------------------------------------

    def _build_indexes(self) -> None:
        page = self._pages["indexes"]
        card = page.add_card()

        self.app_index_label = QLabel("Rescan to refresh", self)
        self.app_index_label.setObjectName("readout")
        self.app_index_label.setFont(mono_font(10))
        apps_btn = QPushButton("Rescan apps", self)
        apps_btn.clicked.connect(lambda: self._rescan("apps"))
        row = card.add_row("Application index", apps_btn,
                           "Installed programs the assistant can launch by name.")
        row.set_description(row.description_label.text())
        card.add(self.app_index_label)

        self.music_index_label = QLabel("Rescan to refresh", self)
        self.music_index_label.setObjectName("readout")
        self.music_index_label.setFont(mono_font(10))
        music_btn = QPushButton("Rescan music", self)
        music_btn.clicked.connect(lambda: self._rescan("music"))
        card.add_row("Music index", music_btn,
                     "Local audio files the assistant can play by name.")
        card.add(self.music_index_label)

    def _rescan(self, kind: str) -> None:
        label = self.app_index_label if kind == "apps" else self.music_index_label
        label.setText("Rescanning...")

        def worker() -> None:
            try:
                if kind == "apps":
                    from actions import app_control
                    result = app_control.rescan_app_index()
                    text = str(result.get("response_text", "App index scan completed."))
                else:
                    from actions import media_control
                    result = media_control.rescan_media_index()
                    text = str(result.get("response_text", "Music index scan completed."))
            except Exception as exc:                       # pragma: no cover
                trace_exception("ui.settings", exc, event=f"rescan_{kind}_failed")
                text = f"Scan failed: {exc}"
            trace_event("ui.settings", f"rescan_{kind}_completed", message=text)
            self._host.log_requested.emit(text)
            # Back onto the GUI thread: this runs on a plain worker thread.
            QTimer.singleShot(0, lambda: label.setText(text))

        threading.Thread(target=worker, daemon=True).start()

    # -- About ------------------------------------------------------------

    def _build_about(self) -> None:
        page = self._pages["about"]
        card = page.add_card()

        llm = self._pipeline.get_llm_status()
        stt = self._pipeline.get_stt_status()
        wake = self._pipeline.get_wakeword_status()

        def readout(text: str) -> QLabel:
            label = QLabel(text, self)
            label.setObjectName("readout")
            label.setFont(mono_font(10))
            label.setWordWrap(True)
            return label

        card.add_row("Assistant", readout(self._name))
        card.add_row("Language model", readout(
            f"{llm.get('state', 'unknown')} ({llm.get('mode', 'n/a')})"))
        card.add_row("Speech to text", readout(
            f"{'ready' if stt.get('initialized') else 'lazy'} on "
            f"{stt.get('device', 'cpu')}"))
        card.add_row("Wake word", readout(
            "available" if wake.get("available") else
            ("starting" if wake.get("initializing") else
             ("enabled, unavailable" if wake.get("enabled") else "disabled"))))
        card.add_row("Vision", readout(
            "supported" if llm.get("supports_vision") else "unsupported"))

    # -- apply ------------------------------------------------------------

    def _apply(self) -> None:
        """Push the buffered pages to the pipeline, in the established order.

        Order matters: set_tts_profile also drives branding and therefore the
        persona theme, and set_compute_mode kicks off an async capability
        rescan whose result the wake-word re-init should not race.
        """
        self._appearance_committed = True
        pipeline = self._pipeline
        host = self._host

        try:
            pipeline.set_tts_enabled(self.tts_check.isChecked())
            profile = str(self.tts_profile_combo.currentData() or "female")
            pipeline.set_tts_profile(profile)
            host._refresh_assistant_branding()

            verbosity = str(self.verbosity_combo.currentData() or "normal")
            pipeline.set_response_verbosity(verbosity)

            pipeline.set_realtime_web_enabled(self.verified_web_check.isChecked())
            host._set_verified_web_controls(pipeline.is_realtime_web_enabled())

            selected_mode = str(self.compute_combo.currentData() or "auto")
            updated = pipeline.set_compute_mode(selected_mode)
            self._report_compute(updated, selected_mode)

            phrases = [self.phrase_list.item(i).text().strip().lower()
                       for i in range(self.phrase_list.count())]
            phrases = [p for p in phrases if p]
            if not phrases:
                phrases = list(DEFAULT_WAKEWORD_CONFIG.get(
                    "activation_phrases", ["jarvis", "hey jarvis"]))

            wake_enabled = self.wake_check.isChecked()
            pipeline.update_wakeword_settings(
                enabled=wake_enabled,
                sensitivity=float(self.wake_sensitivity.value()),
                activation_phrases=phrases,
                strict_phrase_prefix=self.strict_prefix_check.isChecked(),
                auto_restart_after_response=self.auto_restart_check.isChecked(),
                follow_up_after_response=self.follow_up_check.isChecked(),
                follow_up_timeout=int(self.follow_up_timeout.value()),
                max_followup_turns=int(self.follow_up_turns.value()),
            )
            if wake_enabled:
                host.enable_wake_word()
            else:
                host.disable_wake_word()

            update_ui_settings(frameless=bool(self.frameless_check.isChecked()))

            host._set_verified_web_controls(pipeline.is_realtime_web_enabled())
            trace_event("ui.settings", "applied", tts_profile=profile,
                        verbosity=verbosity, compute_mode=selected_mode,
                        wake_enabled=bool(wake_enabled))
        except Exception as exc:
            trace_exception("ui.settings", exc)

    def _report_compute(self, updated: dict, requested: str) -> None:
        """GPU warnings go to the log, not into a modal on top of a modal."""
        caps = dict(updated.get("capabilities") or {})
        mode = str(updated.get("mode", requested))
        if not caps or mode != "gpu":
            return
        if not caps.get("gpu_supported"):
            self._host._append_log(
                "GPU mode selected, but no compatible GPU runtime is available; "
                "continuing on CPU.")
        elif not caps.get("llm_gpu_offload_supported"):
            self._host._append_log(
                "GPU mode is on for supported components, but the local LLM "
                "runtime cannot offload to this GPU; replies run on CPU.")
