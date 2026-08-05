"""OS backend selection.

Action modules import `backend` from here and stay platform-neutral;
everything that touches the OS (COM, pactl, ms-settings, xdg-open, ...)
lives in exactly one backend module per platform.
"""

import sys

if sys.platform == "win32":
    from platform_adapters import windows as backend
else:
    from platform_adapters import linux as backend

__all__ = ["backend"]
