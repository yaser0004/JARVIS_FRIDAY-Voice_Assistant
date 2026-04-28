from __future__ import annotations

from collections import deque

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ui.theme import FONT_BODY, FONT_DISPLAY, FONT_MONO
from ui.widgets.status_bar import MetricsStatusBar


class Sidebar(QWidget):
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
        self.setMinimumWidth(220)
        self.setMaximumWidth(360)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.recent_intents = deque(maxlen=5)
        self.intent_bars: list[QProgressBar] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        self.logo_label = QLabel("JARVIS")
        self.logo_label.setFont(FONT_DISPLAY)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.logo_label)

        self.status_label = QLabel("IDLE")
        self.status_label.setFont(FONT_MONO)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.status_label)

        self.metrics = MetricsStatusBar(self)
        root.addWidget(self.metrics)

        self.intent_toggle_btn = QPushButton("Intent Identifier >")
        self.intent_toggle_btn.setFont(FONT_BODY)
        self.intent_toggle_btn.setCheckable(True)
        self.intent_toggle_btn.setChecked(False)
        self.intent_toggle_btn.toggled.connect(self._toggle_intent_panel)
        root.addWidget(self.intent_toggle_btn)

        self.intent_panel = QFrame(self)
        self.intent_panel.setVisible(False)
        intents_layout = QVBoxLayout(self.intent_panel)
        intents_layout.setContentsMargins(6, 6, 6, 6)
        intents_layout.setSpacing(6)

        intents_title = QLabel("Recent Intents")
        intents_title.setFont(FONT_MONO)
        intents_layout.addWidget(intents_title)

        self.intent_rows: list[QHBoxLayout] = []
        for _ in range(5):
            row = QHBoxLayout()
            name = QLabel("-")
            name.setFont(FONT_MONO)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            row.addWidget(name, 3)
            row.addWidget(bar, 4)
            intents_layout.addLayout(row)
            self.intent_rows.append(row)
            self.intent_bars.append(bar)

        diagnostics_title = QLabel("Top Candidates")
        diagnostics_title.setFont(FONT_MONO)
        intents_layout.addWidget(diagnostics_title)

        self.candidate_rows: list[QLabel] = []
        for _ in range(3):
            lbl = QLabel("-")
            lbl.setFont(FONT_MONO)
            lbl.setWordWrap(False)
            intents_layout.addWidget(lbl)
            self.candidate_rows.append(lbl)

        self.runtime_info_label = QLabel("runtime: -")
        self.runtime_info_label.setFont(FONT_MONO)
        self.runtime_info_label.setWordWrap(True)
        intents_layout.addWidget(self.runtime_info_label)

        catalog_title = QLabel("Supported Intents")
        catalog_title.setFont(FONT_MONO)
        intents_layout.addWidget(catalog_title)

        self.intent_catalog_label = QLabel("-")
        self.intent_catalog_label.setFont(FONT_MONO)
        self.intent_catalog_label.setWordWrap(True)
        self.intent_catalog_label.setText(self._format_intent_catalog())
        intents_layout.addWidget(self.intent_catalog_label)

        model_label = QLabel("Intent Model")
        model_label.setFont(FONT_BODY)
        intents_layout.addWidget(model_label)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(["LinearSVC", "BiLSTM", "DistilBERT"])
        self.model_selector.setCurrentText("DistilBERT")
        intents_layout.addWidget(self.model_selector)

        root.addWidget(self.intent_panel)

        root.addStretch(1)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.settings_btn = QPushButton("")
        self.settings_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.setFixedWidth(40)
        self.settings_btn.setFont(FONT_BODY)
        controls.addWidget(self.settings_btn, 1)

        self.logs_btn = QPushButton("")
        self.logs_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView))
        self.logs_btn.setToolTip("Logs")
        self.logs_btn.setFixedWidth(40)
        self.logs_btn.setFont(FONT_BODY)
        controls.addWidget(self.logs_btn, 1)

        root.addLayout(controls)

    def _toggle_intent_panel(self, expanded: bool) -> None:
        self.intent_panel.setVisible(expanded)
        self.intent_toggle_btn.setText("Intent Identifier v" if expanded else "Intent Identifier >")

    def set_brand_name(self, brand_name: str) -> None:
        normalized_brand = str(brand_name or "JARVIS").strip().upper() or "JARVIS"
        self.logo_label.setText(normalized_brand)

    def set_state(self, state: str) -> None:
        self.status_label.setText(state.upper())

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

    def _format_intent_catalog(self) -> str:
        lines = []
        for key in sorted(self.INTENT_INFO.keys()):
            desc, _ = self.INTENT_INFO[key]
            lines.append(f"- {key}: {desc}")
        return "\n".join(lines)

    def set_intent_diagnostics(self, diagnostics: dict) -> None:
        payload = diagnostics if isinstance(diagnostics, dict) else {}
        candidates = payload.get("top_candidates") if isinstance(payload.get("top_candidates"), list) else []

        for idx, lbl in enumerate(self.candidate_rows):
            if idx < len(candidates) and isinstance(candidates[idx], dict):
                name = str(candidates[idx].get("intent", "-"))
                conf = float(candidates[idx].get("confidence", 0.0))
                desc, example = self.INTENT_INFO.get(name, ("", ""))
                lbl.setText(f"{name}: {int(conf * 100)}%")
                if desc or example:
                    lbl.setToolTip(f"{desc}\nexample: {example}".strip())
                else:
                    lbl.setToolTip("")
            else:
                lbl.setText("-")
                lbl.setToolTip("")

        runtime = str(payload.get("runtime", "") or "-")
        provider = str(payload.get("provider", "") or "-")
        latency = float(payload.get("latency_ms", 0.0) or 0.0)
        self.runtime_info_label.setText(f"runtime: {runtime} | {provider} | {latency:.1f} ms")

