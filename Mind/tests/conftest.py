import importlib.util
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

# The daemon source file uses a hyphen in its name (mind-daemon.py),
# which is not a valid Python module name. Load it explicitly.
_DAEMON_FILE = Path(__file__).resolve().parent.parent / "daemon" / "mind-daemon.py"

_spec = importlib.util.spec_from_file_location("mind_daemon", _DAEMON_FILE)
bd = importlib.util.module_from_spec(_spec)
sys.modules["mind_daemon"] = bd
sys.path.insert(0, str(_DAEMON_FILE.parent))
_spec.loader.exec_module(bd)


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """A BubbleDaemon/Mind daemon with all paths pointed at a temp directory."""
    monkeypatch.setenv("MIND_CTX_SIZE", "8192")
    monkeypatch.setenv("MIND_LLM_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("MIND_SYSTEM_PROMPT", "")
    monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / "mind"))
    monkeypatch.setenv("MIND_MEMORY_SEED_DIR", "")
    monkeypatch.setenv("MIND_MEMORY_SYNC_INDEX", "0")
    # Prevent GLib.idle_add calls from blowing up in tests (no GLib main loop).
    monkeypatch.setattr(bd.GLib, "idle_add", lambda *a, **k: None)

    d = bd.BubbleDaemon()
    d._lock = threading.RLock()
    yield d
