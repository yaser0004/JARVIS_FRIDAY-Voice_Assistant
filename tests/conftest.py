"""Test bootstrap.

`sounddevice` needs a PortAudio system library that CI/headless boxes often
lack, and importing the pipeline pulls it in transitively. A stub keeps the
import graph testable without audio hardware.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - depends on the host
    import sounddevice  # noqa: F401
except Exception:  # pragma: no cover
    stub = types.ModuleType("sounddevice")

    class _InputStream:
        def __init__(self, *args, **kwargs):
            raise OSError("sounddevice stub: no audio device in tests")

    stub.InputStream = _InputStream
    stub.play = lambda *a, **k: None
    stub.stop = lambda *a, **k: None
    stub.default = types.SimpleNamespace(device=None)
    sys.modules["sounddevice"] = stub
