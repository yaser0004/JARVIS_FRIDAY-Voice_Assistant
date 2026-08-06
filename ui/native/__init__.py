"""Native window chrome backend selection.

Mirrors `platform_adapters`: the window imports `WindowChrome` from here and
stays platform-neutral, and everything that touches Win32 lives in exactly
one module.

Named `ui.native` rather than `ui.platform` on purpose -- a module called
`ui.platform` sitting next to the standard library's `platform` is a trap
nobody needs.
"""

import sys

if sys.platform == "win32":
    from ui.native.win_chrome import WindowChrome
else:
    from ui.native.null_chrome import WindowChrome

__all__ = ["WindowChrome"]
