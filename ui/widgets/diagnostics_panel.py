"""Developer diagnostics, hidden by default.

Everything here used to sit permanently in a left sidebar: recent intents
with progress bars, top candidates, a runtime readout, all fourteen
supported intents as a bulleted list, a model picker and raw CPU/GPU/RAM
text. That is an engineer's debug view, and it was the main reason the
window read as an instrument panel rather than as an assistant.

It is the same information, behind one toggle.
"""

from __future__ import annotations

from collections import deque

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtProperty
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import body_font, display_font, manager as theme_manager, mono_font
from ui.widgets.status_bar import MetricsStatusBar


PANEL_WIDTH = 300


class DiagnosticsPanel(QFrame):
    """Slide-out panel. Width animates so it can sit in a normal layout."""

    INTENT_INFO = {
        "launch_app": ("Open or start desktop apps", "can you please launch chrome"),
        "close_app": ("Close or quit running apps", "please close spotify for me"),
        "web_search": ("Search the web for information", "look up latest ai news"),
        "open_website": ("Open URLs and websites", "open github.com"),
        "play_media": ("Play, pause, and control media", "play lofi on youtube"),
        "system_volume": ("Change or query system volume", "do me a favor and set volume to 40"),
        "system_brightness": ("Change or query screen brightness", "set brightness to 65 please"),
        "power_control": ("Power and device toggles", "turn off monitor"),
        "system_settings": ("Open Windows settings pages", "open bluetooth settings"),
        "general_qa": ("General questions and conversation", "explain transformers in nlp"),
        "vision_query": ("Analyze camera, screenshots, images", "what do you see on screen"),
        "file_control": ("Find/open local files", "find my resume pdf"),
        "clipboard_action": ("Read/copy/paste clipboard", "read clipboard"),
        "stop_cancel": ("Stop current assistant action", "cancel that"),
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("diagnosticsPanel")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.recent_intents: deque = deque(maxlen=5)
        self.intent_bars: list[QProgressBar] = []
        self.intent_rows: list[QHBoxLayout] = []
        self.candidate_rows: list[QLabel] = []
        self._open = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)

        body = QWidget(scroll)
        body.setMinimumWidth(PANEL_WIDTH)
        scroll.setWidget(body)

        root = QVBoxLayout(body)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        title = QLabel("Diagnostics")
        title.setObjectName("heading")
        title.setFont(display_font(13, tracking=4.0))
        root.addWidget(title)

        root.addWidget(self._section("System"))
        self.metrics = MetricsStatusBar(self)
        root.addWidget(self.metrics)

        root.addWidget(self._section("Recent intents"))
        for _ in range(5):
            row = QHBoxLayout()
            row.setSpacing(8)
            name = QLabel("-")
            name.setObjectName("readout")
            name.setFont(mono_font(9))
            bar = QProgressBar()
            bar.setObjectName("subtle")
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(5)
            row.addWidget(name, 3)
            row.addWidget(bar, 4)
            root.addLayout(row)
            self.intent_rows.append(row)
            self.intent_bars.append(bar)

        root.addWidget(self._section("Top candidates"))
        for _ in range(3):
            label = QLabel("-")
            label.setObjectName("readout")
            label.setFont(mono_font(9))
            root.addWidget(label)
            self.candidate_rows.append(label)

        self.runtime_info_label = QLabel("runtime: -")
        self.runtime_info_label.setObjectName("readout")
        self.runtime_info_label.setFont(mono_font(9))
        self.runtime_info_label.setWordWrap(True)
        root.addWidget(self.runtime_info_label)

        root.addWidget(self._section("Intent model"))
        self.model_selector = QComboBox(self)
        # BiLSTM has no inference runtime (falls back to LinearSVC), so it is not offered here.
        self.model_selector.addItems(["LinearSVC", "DistilBERT"])
        self.model_selector.setCurrentText("DistilBERT")
        self.model_selector.setFont(body_font(10))
        root.addWidget(self.model_selector)

        root.addWidget(self._section("Supported intents"))
        self.intent_catalog_label = QLabel(self._format_intent_catalog())
        self.intent_catalog_label.setObjectName("caption")
        self.intent_catalog_label.setFont(mono_font(8))
        self.intent_catalog_label.setWordWrap(True)
        root.addWidget(self.intent_catalog_label)

        root.addStretch(1)

        self._anim = QPropertyAnimation(self, b"panelWidth", self)
        self._anim.setDuration(280)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.setFixedWidth(0)
        self.setVisible(False)
        theme_manager().changed.connect(self._apply_fonts)

    # -- helpers ----------------------------------------------------------

    def _section(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("sectionTitle")
        label.setFont(mono_font(8, tracking=14.0))
        return label

    def _format_intent_catalog(self) -> str:
        return "\n".join(
            f"{key}  {self.INTENT_INFO[key][0]}" for key in sorted(self.INTENT_INFO)
        )

    def _apply_fonts(self) -> None:
        self.runtime_info_label.setFont(mono_font(9))
        self.intent_catalog_label.setFont(mono_font(8))

    # -- open/close -------------------------------------------------------

    @pyqtProperty(int)
    def panelWidth(self) -> int:
        return self.width()

    @panelWidth.setter
    def panelWidth(self, value: int) -> None:
        self.setFixedWidth(max(0, int(value)))
        if self.width() <= 0 and self.isVisible():
            self.setVisible(False)

    def set_open(self, opened: bool, animate: bool = True) -> None:
        opened = bool(opened)
        if opened == self._open:
            return
        self._open = opened

        self._anim.stop()
        if opened:
            self.setVisible(True)
        if not animate:
            self.panelWidth = PANEL_WIDTH if opened else 0
            self.setVisible(opened)
            return
        self._anim.setStartValue(self.width())
        self._anim.setEndValue(PANEL_WIDTH if opened else 0)
        self._anim.start()

    def toggle(self) -> bool:
        self.set_open(not self._open)
        return self._open

    @property
    def is_open(self) -> bool:
        return self._open

    # -- data -------------------------------------------------------------

    def add_intent(self, intent: str, confidence: float) -> None:
        self.recent_intents.appendleft((intent, confidence))
        items = list(self.recent_intents)

        for idx, row in enumerate(self.intent_rows):
            name_label = row.itemAt(0).widget()
            bar = row.itemAt(1).widget()
            if idx < len(items):
                item_intent, conf = items[idx]
                name_label.setText(item_intent)
                bar.setValue(int(conf * 100))
            else:
                name_label.setText("-")
                bar.setValue(0)

    def set_intent_diagnostics(self, diagnostics: dict) -> None:
        payload = diagnostics if isinstance(diagnostics, dict) else {}
        candidates = payload.get("top_candidates")
        candidates = candidates if isinstance(candidates, list) else []

        for idx, label in enumerate(self.candidate_rows):
            if idx < len(candidates) and isinstance(candidates[idx], dict):
                name = str(candidates[idx].get("intent", "-"))
                conf = float(candidates[idx].get("confidence", 0.0))
                desc, example = self.INTENT_INFO.get(name, ("", ""))
                label.setText(f"{name}  {int(conf * 100)}%")
                label.setToolTip(f"{desc}\nexample: {example}".strip() if desc or example else "")
            else:
                label.setText("-")
                label.setToolTip("")

        runtime = str(payload.get("runtime", "") or "-")
        provider = str(payload.get("provider", "") or "-")
        latency = float(payload.get("latency_ms", 0.0) or 0.0)
        self.runtime_info_label.setText(
            f"runtime {runtime}  |  {provider}  |  {latency:.1f} ms"
        )
