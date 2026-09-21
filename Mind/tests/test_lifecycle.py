"""
Tests for the Mind lifecycle / MX telemetry helpers.

The MX is how the mind "feels" its own world: idle phase, context window
usage, and how long it / the user has been quiet. These are pure helpers on
the daemon, exercised directly.
"""

import time
import pytest
from conftest import bd


class TestIdlePhase:
    def test_active(self, daemon):
        assert daemon._idle_phase(0.5) == "active"

    def test_idle(self, daemon):
        assert daemon._idle_phase(2) == "idle"

    def test_waiting(self, daemon):
        assert daemon._idle_phase(10) == "waiting"

    def test_quiet(self, daemon):
        assert daemon._idle_phase(60) == "quiet"


class TestIdleLabel:
    def test_just_now(self, daemon):
        assert daemon._idle_label(0) == "just now"

    def test_minutes(self, daemon):
        assert daemon._idle_label(5) == "5m ago"

    def test_hours(self, daemon):
        assert daemon._idle_label(120) == "2.0h ago"


class TestContextPercent:
    def test_empty_history_is_zero(self, daemon):
        assert daemon._ctx_percent() == 0.0

    def test_caps_at_100(self, daemon):
        # ~4 chars per token, so 8192 * 4 = ~32768 chars fills the window.
        daemon._history = [{"role": "user", "content": "x" * 40000}]
        assert daemon._ctx_percent() == 100.0

    def test_reports_partial_usage(self, daemon):
        daemon._ctx_size = 8192
        daemon._history = [{"role": "user", "content": "x" * 800}]
        pct = daemon._ctx_percent()
        assert 0.0 < pct < 100.0


class TestTelemetry:
    def test_returns_keys(self, daemon):
        tel = daemon._telemetry()
        for key in ("ctx_pct", "ctx_total", "token_est",
                    "since_user", "since_assistant", "idle_phase", "nudge_count"):
            assert key in tel


class TestHistoryTrim:
    def test_trims_over_max(self, daemon):
        # Force MAX_HISTORY_TURNS low and add extra turns.
        bd.MAX_HISTORY_TURNS = 5
        daemon._history = [
            {"role": "user", "content": "x"} for _ in range(10)
        ]
        daemon._trim_history()
        assert len(daemon._history) == 5
