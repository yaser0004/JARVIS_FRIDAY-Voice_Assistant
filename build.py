"""Package JARVIS with Nuitka.

Windows produces JARVIS.exe; on Linux the same command produces an ELF binary
(the --windows-* flags are only emitted on Windows).
"""

import subprocess
import sys

COMMON_FLAGS = [
    "--standalone",
    "--onefile",
    "--output-dir=dist",
    "--enable-plugin=pyqt6",
    "--include-data-dir=assets=assets",
    "--include-data-dir=ml/models=ml/models",
    "--include-data-dir=models=models",
    # TensorFlow is training-only (requirements-train.txt) and must not be bundled.
    "--include-package=sklearn",
    "--include-package=transformers",
    "--include-package=onnxruntime",
    "--include-package=spacy",
    "--include-package=faster_whisper",
    "--include-package=openwakeword",
    "--include-package=kokoro",
    "--include-package=chromadb",
    "--include-package=litert_lm",
    "--assume-yes-for-downloads",
]

WINDOWS_FLAGS = [
    "--windows-disable-console",
    "--windows-icon-from-ico=assets/icon.ico",
    "--output-filename=JARVIS.exe",
    "--windows-uac-admin",
]

LINUX_FLAGS = [
    "--output-filename=jarvis",
    "--linux-icon=assets/icon.ico",
]


def build_command() -> list[str]:
    platform_flags = WINDOWS_FLAGS if sys.platform == "win32" else LINUX_FLAGS
    return [sys.executable, "-m", "nuitka", *COMMON_FLAGS, *platform_flags, "main.py"]


if __name__ == "__main__":
    subprocess.run(build_command(), check=True)
    artifact = "JARVIS.exe" if sys.platform == "win32" else "jarvis"
    print(f"\n{artifact} built successfully in dist/")
