"""Frame-rate benchmark for the animated UI.

Two numbers matter and they are not the same:

  per-widget paint  how long one widget's paintEvent takes
  delivered frame   how far apart frames actually arrive

A widget can sit comfortably inside its paint budget while the composed
window still delivers 32 FPS, because the orb, the waves, the chat
animations and the window present all land in the same frame. The delivered
interval is the number that means 60 FPS; the per-widget number only tells
you which widget to blame when it fails.

Usage:
    python qa/ui_frame_bench.py
    python qa/ui_frame_bench.py --seconds 4 --json report.json
    python qa/ui_frame_bench.py --size 1600x1000

Exits non-zero if any budget is missed, so it can gate a change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication


# Budgets in milliseconds.
DELIVERED_P95_BUDGET = 18.0   # 60 FPS with headroom for one slow frame
ORB_PAINT_P95_BUDGET = 8.0    # at the 64px size the orb ships at

STATES = ["IDLE", "LISTENING", "PROCESSING", "SPEAKING"]

# The clock realigns its phase within the first second after the window is
# exposed. Measuring through that transient reports a startup artifact
# rather than steady state.
WARMUP_MS = 2000


def parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1180x740")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=2.5,
                        help="measurement window per state (default 2.5)")
    parser.add_argument("--size", type=parse_size, default=(1180, 740),
                        help="window size, e.g. 1600x1000")
    parser.add_argument("--theme", default=None,
                        help="pin a theme preset (jarvis, friday, monochrome, daylight)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the full report to this path")
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])

    from ui import anim_clock
    from ui.main_window import JarvisMainWindow
    from ui.theme import manager, theme

    if args.theme:
        manager().initialize()
        manager().set_preset(args.theme, persist=False)

    window = JarvisMainWindow()
    window.resize(*args.size)

    # A populated transcript, an expanded panel and a live waveform are the
    # realistic worst case, not an empty window.
    for i in range(14):
        window.chat.add_message(
            "user" if i % 2 == 0 else "assistant",
            f"line {i}: sustained conversation content to populate the transcript",
        )
    # The old left sidebar is gone; the same diagnostics content now slides out
    # from the right behind this toggle.
    window.app_bar.diagnostics_btn.setChecked(True)
    window.show()
    window.orb.set_profiling(True)

    report: dict = {
        "theme": theme().name,
        "window": f"{args.size[0]}x{args.size[1]}",
        "orb": f"{window.orb.width()}x{window.orb.height()}",
        "states": {},
    }
    step = {"i": 0}

    def run_state() -> None:
        if step["i"] > 0:
            state = STATES[step["i"] - 1]
            report["states"][state] = anim_clock.clock().stats()

        if step["i"] >= len(STATES):
            finish()
            return

        state = STATES[step["i"]]
        step["i"] += 1
        window.orb.set_state(state)
        # Manual amplitude override: this machine may have no audio input, and
        # a silent orb would measure a load nobody actually sees.
        window.orb.set_audio_level(0.6 if state in ("LISTENING", "SPEAKING") else 0.0)
        anim_clock.clock().reset_stats()
        QTimer.singleShot(int(args.seconds * 1000), run_state)

    def finish() -> None:
        report["orb_paint"] = window.orb.paint_stats()
        report["orb"] = f"{window.orb.width()}x{window.orb.height()}"

        print(f"theme {report['theme']}   window {report['window']}   "
              f"orb {report['orb']}\n")
        print(f"{'state':<12} {'p50':>8} {'p95':>8} {'p99':>8} {'fps':>7}")
        worst = 0.0
        for state, stats in report["states"].items():
            if not stats.get("samples"):
                continue
            worst = max(worst, stats["p95_ms"])
            print(f"{state:<12} {stats['p50_ms']:>8.2f} {stats['p95_ms']:>8.2f} "
                  f"{stats['p99_ms']:>8.2f} {stats['mean_fps']:>7.1f}")

        paint = report["orb_paint"]
        print(f"\norb paint    p50 {paint.get('p50_ms', 0):.2f} ms   "
              f"p95 {paint.get('p95_ms', 0):.2f} ms   "
              f"quality {paint.get('quality', 1.0)}")

        delivered_ok = worst <= DELIVERED_P95_BUDGET
        paint_ok = paint.get("p95_ms", 0.0) <= ORB_PAINT_P95_BUDGET
        print(f"\ndelivered p95 {worst:.2f} ms  (budget {DELIVERED_P95_BUDGET}) "
              f"{'PASS' if delivered_ok else 'FAIL'}")
        print(f"orb paint p95 {paint.get('p95_ms', 0):.2f} ms  "
              f"(budget {ORB_PAINT_P95_BUDGET}) {'PASS' if paint_ok else 'FAIL'}")

        report["verdict"] = "PASS" if (delivered_ok and paint_ok) else "FAIL"
        report["realigns"] = anim_clock.clock()._realigns
        if args.json:
            args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nreport written to {args.json}")

        window.waveform.set_state("IDLE")
        window.tray.hide()
        app.exit(0 if report["verdict"] == "PASS" else 1)

    QTimer.singleShot(WARMUP_MS, run_state)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
