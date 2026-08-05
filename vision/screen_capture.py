from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


class ScreenCapture:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or (Path.cwd() / "data" / "captures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _grab(self):
        # pyautogui covers Windows and X11; on Wayland it typically fails,
        # so fall back to gnome-screenshot which goes through the portal.
        try:
            import pyautogui

            return pyautogui.screenshot()
        except Exception as exc:
            grab_error = exc

        if shutil.which("gnome-screenshot"):
            tmp_path = self.output_dir / f"grab_{int(time.time() * 1000)}.png"
            try:
                result = subprocess.run(
                    ["gnome-screenshot", "-f", str(tmp_path)],
                    capture_output=True,
                    timeout=15,
                )
                if result.returncode == 0 and tmp_path.exists():
                    from PIL import Image

                    with Image.open(tmp_path) as img:
                        image = img.copy()
                    return image
            finally:
                tmp_path.unlink(missing_ok=True)

        raise RuntimeError(f"screen capture failed: {grab_error}")

    def capture_full(self, save: bool = True) -> Tuple[object, Optional[Path]]:
        image = self._grab()
        saved_path = None
        if save:
            saved_path = self.output_dir / f"screen_{int(time.time() * 1000)}.png"
            image.save(saved_path)
        return image, saved_path

    def capture_region(
        self, x: int, y: int, width: int, height: int, save: bool = True
    ) -> Tuple[object, Optional[Path]]:
        image = self._grab().crop((x, y, x + width, y + height))
        saved_path = None
        if save:
            saved_path = self.output_dir / f"region_{int(time.time() * 1000)}.png"
            image.save(saved_path)
        return image, saved_path
