"""
Dev launcher: watches sections/, services/, utils/ and touches app.py on .py change.
Run: python run_dev.py
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Install watchdog: pip install watchdog")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
APP_PY = PROJECT_ROOT / "app.py"
WATCH_DIRS = ["sections", "services", "utils"]


class TouchAppPyHandler(FileSystemEventHandler):
    """On any .py change in watched dirs, touch app.py so Streamlit reruns."""

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".py"):
            try:
                APP_PY.touch()
            except Exception:
                pass


def run_watcher():
    observer = Observer()
    for d in WATCH_DIRS:
        path = PROJECT_ROOT / d
        if path.exists():
            observer.schedule(TouchAppPyHandler(), str(path), recursive=True)
    observer.start()
    try:
        while True:
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main():
    watcher = threading.Thread(target=run_watcher, daemon=True)
    watcher.start()
    os.chdir(PROJECT_ROOT)
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.runOnSave", "true",
        ],
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
