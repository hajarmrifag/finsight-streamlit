"""Shared constants and constructors for the Streamlit app tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / 'app.py')

# These limits exist to catch hangs, not to measure speed. A fresh environment can spend 20+ seconds
# loading native libraries on its first run (seen on macOS right after creating the virtualenv), so a
# tight limit turns first-run cost into a false failure.
APP_TIMEOUT = 60
SEMANTIC_TIMEOUT = 180  # includes the one-time MiniLM download on a clean machine


def new_app():
    """An AppTest for app.py whose runs, including follow-up interactions, share APP_TIMEOUT."""
    return AppTest.from_file(APP, default_timeout=APP_TIMEOUT)
