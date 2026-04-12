from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor, QFont, QFontDatabase


CYAN = QColor(0, 245, 255)
CYAN_DIM = QColor(0, 184, 200)
CYAN_GHOST = QColor(0, 245, 255, 18)
ORANGE = QColor(255, 107, 43)
GREEN = QColor(0, 255, 136)
YELLOW = QColor(255, 215, 0)
RED = QColor(255, 51, 85)
BG_MAIN = QColor(3, 10, 15)
BG_PANEL = QColor(6, 15, 24, 220)
BORDER = QColor(0, 245, 255, 46)
TEXT = QColor(200, 234, 240)
TEXT_DIM = QColor(90, 138, 150)


def load_fonts() -> None:
    root = Path(__file__).resolve().parent.parent
    font_paths = [
        root / "assets" / "fonts" / "Orbitron-Bold.ttf",
        root / "assets" / "fonts" / "ShareTechMono-Regular.ttf",
    ]
    for path in font_paths:
        if path.exists() and path.stat().st_size > 0:
            QFontDatabase.addApplicationFont(str(path))


FONT_DISPLAY = QFont("Orbitron", 10, QFont.Weight.Bold)
FONT_MONO = QFont("Share Tech Mono", 10)
FONT_BODY = QFont("Rajdhani", 11)
