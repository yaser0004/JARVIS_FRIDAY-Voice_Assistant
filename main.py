import multiprocessing
import os
import sys
import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from core.session_logging import SessionTerminalLogger, trace_event, trace_exception
from ui.main_window import JarvisMainWindow


def _run() -> int:
    session_logger = SessionTerminalLogger()
    trace_event(
        "app.lifecycle",
        "startup_begin",
        pid=os.getpid(),
        cwd=os.getcwd(),
        argv=list(sys.argv),
    )
    multiprocessing.freeze_support()

    # Prevent common OpenMP runtime clashes between ML/audio dependencies on Windows.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    trace_event(
        "app.lifecycle",
        "runtime_env_configured",
        jarvis_dev=os.getenv("JARVIS_DEV", ""),
        omp_threads=os.getenv("OMP_NUM_THREADS", ""),
        openblas_threads=os.getenv("OPENBLAS_NUM_THREADS", ""),
        mkl_threads=os.getenv("MKL_NUM_THREADS", ""),
    )

    if os.getenv("JARVIS_DEV", "").strip().lower() in {"1", "true", "yes"}:
        import builtins

        original_import = builtins.__import__
        torch_import_logged = {"done": False}
        st_import_logged = {"done": False}

        def debug_import(name, globals=None, locals=None, fromlist=(), level=0):
            if not st_import_logged["done"] and (
                name == "sentence_transformers" or name.startswith("sentence_transformers.")
            ):
                st_import_logged["done"] = True
                print(f"JARVIS debug: importing {name}", flush=True)

            if not torch_import_logged["done"] and (name == "torch" or name.startswith("torch.")):
                torch_import_logged["done"] = True
                print(f"JARVIS debug: importing {name}", flush=True)
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = debug_import

    print("JARVIS startup: initializing QApplication...", flush=True)
    trace_event("app.lifecycle", "qapplication_init_start")

    try:
        app = QApplication(sys.argv)
        trace_event("app.lifecycle", "qapplication_init_done")
        app.setApplicationName("JARVIS")
        app.setOrganizationName("JARVIS")
        if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
            app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

        with open("assets/style.qss", "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
        trace_event("app.lifecycle", "stylesheet_loaded", path="assets/style.qss")

        print("JARVIS startup: creating main window...", flush=True)
        trace_event("app.lifecycle", "main_window_create_start")
        window = JarvisMainWindow()
        window.show()
        trace_event("app.lifecycle", "main_window_visible")
        print("JARVIS startup: starting pipeline...", flush=True)
        window.start_pipeline()
        trace_event("app.lifecycle", "pipeline_start_requested")

        print("JARVIS startup: entering event loop.", flush=True)
        trace_event("app.lifecycle", "event_loop_enter")
        exit_code = app.exec()
        trace_event("app.lifecycle", "event_loop_exit", exit_code=exit_code)
        return exit_code
    except Exception as exc:
        trace_exception("app.lifecycle", exc)
        traceback.print_exc()
        raise
    finally:
        trace_event("app.lifecycle", "shutdown_begin")
        session_path = session_logger.stop()
        print(f"JARVIS session log saved: {session_path}", flush=True)
        print(f"JARVIS trace log saved: {session_logger.trace_path}", flush=True)


if __name__ == "__main__":
    sys.exit(_run())


