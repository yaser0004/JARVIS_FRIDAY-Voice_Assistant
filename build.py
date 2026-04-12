import subprocess
import sys

NUITKA_CMD = [
    sys.executable,
    "-m",
    "nuitka",
    "--standalone",
    "--onefile",
    "--windows-disable-console",
    "--windows-icon-from-ico=assets/icon.ico",
    "--output-filename=JARVIS.exe",
    "--output-dir=dist",
    "--enable-plugin=pyqt6",
    "--include-data-dir=assets=assets",
    "--include-data-dir=ml/models=ml/models",
    "--include-data-dir=models=models",
    "--include-package=sklearn",
    "--include-package=tensorflow",
    "--include-package=transformers",
    "--include-package=onnxruntime",
    "--include-package=spacy",
    "--include-package=faster_whisper",
    "--include-package=openwakeword",
    "--include-package=kokoro",
    "--include-package=chromadb",
    "--include-package=llama_cpp",
    "--windows-uac-admin",
    "--assume-yes-for-downloads",
    "main.py",
]

if __name__ == "__main__":
    subprocess.run(NUITKA_CMD, check=True)
    print("\nJARVIS.exe built successfully in dist/")

