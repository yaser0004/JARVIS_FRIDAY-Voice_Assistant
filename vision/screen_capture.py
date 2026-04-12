from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import pyautogui


class ScreenCapture:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or (Path.cwd() / "data" / "captures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_full(self, save: bool = True) -> Tuple[object, Optional[Path]]:
        image = pyautogui.screenshot()
        saved_path = None
        if save:
            saved_path = self.output_dir / f"screen_{int(time.time() * 1000)}.png"
            image.save(saved_path)
        return image, saved_path

    def capture_region(
        self, x: int, y: int, width: int, height: int, save: bool = True
    ) -> Tuple[object, Optional[Path]]:
        image = pyautogui.screenshot(region=(x, y, width, height))
        saved_path = None
        if save:
            saved_path = self.output_dir / f"region_{int(time.time() * 1000)}.png"
            image.save(saved_path)
        return image, saved_path
