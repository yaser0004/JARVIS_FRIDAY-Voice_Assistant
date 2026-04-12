from __future__ import annotations

import time
from pathlib import Path
from typing import Optional


class WebcamCapture:
    def __init__(self, camera_index: int = 0, output_dir: Path | None = None) -> None:
        self.camera_index = camera_index
        self.output_dir = output_dir or (Path.cwd() / "data" / "captures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_frame(self) -> tuple[bool, Optional[Path], str]:
        try:
            import cv2
        except Exception as exc:
            return False, None, f"OpenCV not available: {exc}"

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return False, None, "Unable to access webcam."

        ok, frame = cap.read()
        cap.release()
        if not ok:
            return False, None, "Could not read webcam frame."

        path = self.output_dir / f"webcam_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(path), frame)
        return True, path, "Frame captured successfully."
