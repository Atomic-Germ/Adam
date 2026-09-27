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

# The letter flow would phone a live llama server when the daemon boots, so the
# fixtures stub it out. Tests that exercise the letter call the real one.
bd.real_parent_letter = bd.BubbleDaemon._maybe_parent_letter


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """A BubbleDaemon/Mind daemon with all paths pointed at a temp directory."""
    monkeypatch.setenv("MIND_CTX_SIZE", "8192")
    monkeypatch.setenv("MIND_LLM_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("MIND_LLM_URL2", "")
    monkeypatch.setenv("MIND_LLM2_URL", "")
    monkeypatch.setenv("MIND_SYSTEM_PROMPT", "")
    monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / "mind"))
    monkeypatch.setenv("MIND_MEMORY_SEED_DIR", "")
    monkeypatch.setenv("MIND_MEMORY_SYNC_INDEX", "0")
    monkeypatch.setenv("MIND_HISTORY_FILE", str(tmp_path / "history.json"))
    monkeypatch.setenv("MIND_SLEEP_KEEP_TURNS", "4")
    # Prevent GLib.idle_add calls from blowing up in tests (no GLib main loop).
    monkeypatch.setattr(bd.GLib, "idle_add", lambda *a, **k: None)
    # The letter flow would phone a live llama server; tests drive it directly.
    monkeypatch.setattr(bd.BubbleDaemon, "_maybe_parent_letter",
                        lambda self: None)

    d = bd.BubbleDaemon()
    d._lock = threading.RLock()
    yield d
    if d._memory is not None:
        d._memory.shutdown()


@pytest.fixture
def two_brain_daemon(tmp_path, monkeypatch):
    """A daemon with a second occupant's brain wired (MIND_LLM_URL2 set).

    The plain `daemon` fixture deliberately leaves the second seat empty so
    the existing single-occupant tests stay isolated.
    """
    monkeypatch.setenv("MIND_CTX_SIZE", "8192")
    monkeypatch.setenv("MIND_LLM_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("MIND_LLM_URL2", "http://127.0.0.1:9998")
    monkeypatch.setenv("MIND_OCCUPANT2_ID", "second")
    monkeypatch.setenv("MIND_SYSTEM_PROMPT", "")
    monkeypatch.setenv("MIND_SYSTEM_PROMPT2", "")
    monkeypatch.setenv("MIND_MEMORY_DIR", str(tmp_path / "mind2"))
    monkeypatch.setenv("MIND_MEMORY_SEED_DIR", "")
    monkeypatch.setenv("MIND_MEMORY_SYNC_INDEX", "0")
    monkeypatch.setenv("MIND_HISTORY_FILE", str(tmp_path / "history2.json"))
    monkeypatch.setenv("MIND_SLEEP_KEEP_TURNS", "4")
    monkeypatch.setattr(bd.GLib, "idle_add", lambda *a, **k: None)
    monkeypatch.setattr(bd.BubbleDaemon, "_maybe_parent_letter",
                        lambda self: None)

    d = bd.BubbleDaemon()
    d._lock = threading.RLock()
    yield d
    if d._memory is not None:
        d._memory.shutdown()
