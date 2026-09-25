"""
Tests for the dream / slow-wave sleep cycle.

A child does not know when it is tired: sleep is not scheduled and the model
does not choose it. Crossing a simple context-pressure threshold (clamped to
60-75%) triggers a dream pass — a direct replay / compaction of the context
window. The dream is embedded into long-term memory, history is wiped, and the
fresh context window opens on nothing but the dream summary.
"""

import threading
import time
from unittest import mock

import pytest
from conftest import bd


class TestSleepConfig:
    def test_default_threshold(self, daemon):
        assert daemon._sleep_ctx_pct == 70.0

    def test_threshold_clamped_low(self, daemon):
        assert daemon._clamp_sleep_pct("10") == 60.0

    def test_threshold_clamped_high(self, daemon):
        assert daemon._clamp_sleep_pct("95") == 75.0

    def test_threshold_in_range(self, daemon):
        assert daemon._clamp_sleep_pct("64") == 64.0

    def test_threshold_garbage_falls_back(self, daemon):
        assert daemon._clamp_sleep_pct("banana") == 70.0

    def test_env_threshold_honoured(self, daemon, monkeypatch):
        monkeypatch.setenv("MIND_SLEEP_CTX_PCT", "63")
        d = bd.BubbleDaemon()
        assert d._sleep_ctx_pct == 63.0
        assert d._sleep_min_turns == bd.DEFAULT_SLEEP_MIN_TURNS


class TestCheckSleep:
    def test_no_trigger_below_threshold(self, daemon):
        daemon._sleep_min_turns = 1
        daemon._history = [{"role": "user", "content": "x" * 100}]
        daemon._start_dream = mock.Mock()
        daemon._check_sleep()
        daemon._start_dream.assert_not_called()

    def test_trigger_above_threshold(self, daemon):
        daemon._sleep_min_turns = 1
        # ~4 chars per token; 8192-token window needs ~32768 chars to hit 100%.
        daemon._sleep_ctx_pct = 60.0
        daemon._history = [{"role": "user", "content": "y" * 4000}] * 8
        daemon._start_dream = mock.Mock()
        daemon._check_sleep()
        daemon._start_dream.assert_called_once()

    def test_does_not_trigger_while_sleeping(self, daemon):
        daemon._sleep_min_turns = 1
        daemon._sleeping = True
        daemon._history = [{"role": "user", "content": "y" * 4000}] * 8
        daemon._start_dream = mock.Mock()
        daemon._check_sleep()
        daemon._start_dream.assert_not_called()

    def test_respects_min_turns_gate(self, daemon):
        daemon._sleep_min_turns = 10
        daemon._history = [{"role": "user", "content": "y" * 4000}] * 2
        daemon._start_dream = mock.Mock()
        daemon._check_sleep()
        daemon._start_dream.assert_not_called()

    def test_no_dream_without_memory(self, daemon):
        daemon._memory = None
        daemon._sleep_min_turns = 1
        daemon._history = [{"role": "user", "content": "y" * 4000}] * 8
        daemon._start_dream = mock.Mock()
        daemon._check_sleep()
        daemon._start_dream.assert_not_called()


class TestDreamPass:
    def test_wipes_history_and_embeds_summary(self, daemon):
        daemon._memory.add_experience("initial memory", source="instruction")
        daemon._history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        with mock.patch.object(
            daemon, "_replay_and_compress", return_value="The dream: keep the letter."
        ):
            daemon._dream_pass(reason="context-pressure")
        assert daemon._dream_summary == "The dream: keep the letter."
        assert daemon._history == []
        assert daemon._dream_count == 1
        assert daemon._sleeping is False
        assert daemon._memory.has_dream() is True

    def test_dream_failure_resets_sleeping(self, daemon):
        daemon._sleeping = True
        daemon._history = [{"role": "user", "content": "x"}]
        with mock.patch.object(
            daemon, "_replay_and_compress", side_effect=RuntimeError("boom")
        ):
            daemon._dream_pass(reason="context-pressure")
        assert daemon._sleeping is False
        assert daemon._history == [{"role": "user", "content": "x"}]

    def test_empty_dream_resets_sleeping(self, daemon):
        daemon._history = [{"role": "user", "content": "x"}]
        with mock.patch.object(daemon, "_replay_and_compress", return_value="   "):
            daemon._dream_pass(reason="context-pressure")
        assert daemon._sleeping is False
        assert daemon._history != []


class TestFirstDream:
    def test_first_dream_fires_with_seeded_memory(self, daemon):
        daemon._memory.add_instruction("The letter: do exactly the one thing told.")
        daemon._start_dream = mock.Mock()
        daemon._maybe_first_dream()
        daemon._start_dream.assert_called_once_with(reason="first-memory")

    def test_first_dream_skipped_after_sleeping(self, daemon):
        daemon._memory.add_instruction("The letter.")
        daemon._memory.add_dream("We already dreamed once.")
        daemon._start_dream = mock.Mock()
        daemon._maybe_first_dream()
        daemon._start_dream.assert_not_called()

    def test_first_dream_skipped_without_memory(self, daemon):
        daemon._memory = None
        daemon._start_dream = mock.Mock()
        daemon._maybe_first_dream()
        daemon._start_dream.assert_not_called()

    def test_first_dream_skipped_without_seed(self, daemon):
        daemon._start_dream = mock.Mock()
        daemon._maybe_first_dream()
        daemon._start_dream.assert_not_called()


class TestDreamBlockInContext:
    def test_context_opens_with_dream_recall(self, daemon):
        daemon._dream_summary = "I dreamed of the first letter."
        daemon.SetSystemPrompt("mind")
        content = daemon._build_system_content()
        assert "--- Dream recall ---" in content
        assert "I dreamed of the first letter." in content

    def test_no_dream_block_when_never_slept(self, daemon):
        daemon.SetSystemPrompt("mind")
        content = daemon._build_system_content()
        assert "Dream recall" not in content

    def test_lock_is_reentrant_build_system_content(self, daemon):
        # Regression: _stream/_clock_tick hold _lock while building the system
        # content, which re-acquires the same lock (dream recall + telemetry).
        # A plain threading.Lock deadlocks the main GLib thread, freezing all
        # D-Bus traffic. The daemon lock must be reentrant.
        daemon._dream_summary = "reentrant dream"
        daemon.SetSystemPrompt("mind")
        with daemon._lock:  # exactly what _stream() does before _build_system_content
            content = daemon._build_system_content()
        assert "--- Dream recall ---" in content

    def test_send_message_runs_stream_off_main_thread(self, daemon):
        # Regression: SendMessage must hand the stream to a background thread;
        # if it ran _stream inline on the GLib main loop, the blocking HTTP
        # call would freeze D-Bus dispatch and signal emission for the entire
        # (minutes-long) generation.
        entered = threading.Event()
        release = threading.Event()

        def slow_stream(*_a, **_k):
            entered.set()
            release.wait(3)

        with mock.patch.object(daemon, "_stream", autospec=True,
                               side_effect=slow_stream):
            start = time.monotonic()
            daemon.SendMessage("hello", "tok-rl", "{}")
        assert entered.wait(0.5) is True, "stream must start"
        assert release.wait(0.5) is False, "stream must still be running"
        assert time.monotonic() - start < 1.0, "SendMessage must return immediately"
        release.set()