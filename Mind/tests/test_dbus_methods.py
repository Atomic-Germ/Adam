"""
Tests for the D-Bus entry points on the Mind daemon (skeleton slice).

These methods form the public surface the GNOME extension calls over D-Bus.
No real D-Bus connection is needed — we instantiate the daemon directly and
call the methods (they are plain Python methods that pydbus exposes).

Memory / memory-selection / scratch / nudge protocol methods are omitted:
those belong to the Arthur memory pieces that come later.
"""

import json

import pytest
from conftest import bd


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------

class TestPing:
    def test_returns_pong(self, daemon):
        assert daemon.Ping() == "pong"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_set_and_get(self, daemon):
        daemon.SetSystemPrompt("you are a helpful, concise mind")
        assert daemon.GetSystemPrompt() == "you are a helpful, concise mind"

    def test_empty_string_clears_prompt(self, daemon):
        daemon.SetSystemPrompt("initial prompt")
        assert daemon.GetSystemPrompt() == "initial prompt"
        daemon.SetSystemPrompt("")
        assert daemon.GetSystemPrompt() == ""

    def test_long_prompt(self, daemon):
        long = "x" * 10000
        daemon.SetSystemPrompt(long)
        assert daemon.GetSystemPrompt() == long


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_gethistory_starts_empty(self, daemon):
        assert json.loads(daemon.GetHistory()) == []

    def test_clearhistory_clears(self, daemon):
        daemon.ClearHistory()
        assert json.loads(daemon.GetHistory()) == []


# ---------------------------------------------------------------------------
# Nudge config
# ---------------------------------------------------------------------------

class TestNudgeConfig:
    def test_set_and_enabled(self, daemon):
        daemon.SetNudgeConfig(True, 20)
        assert daemon._nudge_enabled is True
        assert daemon._nudge_idle_minutes == 20

    def test_idle_minutes_clamped_low(self, daemon):
        daemon.SetNudgeConfig(True, -5)
        assert daemon._nudge_idle_minutes == 1

    def test_idle_minutes_clamped_high(self, daemon):
        daemon.SetNudgeConfig(True, 9999)
        assert daemon._nudge_idle_minutes == 480

    def test_disable(self, daemon):
        daemon.SetNudgeConfig(False, 30)
        assert daemon._nudge_enabled is False
