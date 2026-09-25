#!/usr/bin/env python3
"""
Mind daemon — the "MX" (Model's Experience) for the Mind body.

A lean skeleton port of Bubble's proven daemon pattern, re-wired to run
against the local patched `llama.cpp` OpenAI-compatible server at
http://127.0.0.1:9999 instead of FastFlowLM.

It deliberately carries only:

  * rolling conversation history,
  * the system-prompt framing (the mind's identity/values),
  * live context + telemetry (the mind "feeling" its own world),
  * the model-initiated (nudge) loop and clock tick.

Memory / RAG / embeddings and dream consolidation are Arthur pieces
    ported in (mind_memory.py plus the sleep cycle here). Scratch working-notes,
    the self-prompt revision protocol, nap and the slow-wave night cycle are
    intentionally OMITTED for now.

    Sleep is not scheduled and the model does not choose it. A simple context
    window threshold (60-75%, MIND_SLEEP_CTX_PCT) triggers a dream pass — a
    child does not know it is tired. The dream replays the context window,
    compresses it (keep what is interesting / newly learned / repeated, drop
    the rest), embeds that summary into the same memory space as everything
    else, then wakes with a fresh empty context window that opens on the dream
    summary — identity is re-grounded from long-term memory afterwards.

Wire contract (identical to Bubble's, so the port is mechanical):

  POST  http://127.0.0.1:9999/v1/chat/completions   {"model","messages","stream":true}
  GET   http://127.0.0.1:9999/v1/models             -> detect a model id
  POST  .../v1/chat/completions  {"stream":false}   (clock tick, non-streaming)

llama.cpp emits the exact SSE deltas Bubble already consumes:
  choices[].delta.content            choices[].delta.reasoning_content

D-Bus object:  com.mind.Daemon   at   /com/mind/Daemon   (session bus)
The GNOME extension proxies to it.
"""

import os
import json
import time
import logging
import subprocess
import threading
import datetime
from pathlib import Path

from gi.repository import GLib
from pydbus import SessionBus
from pydbus.generic import signal
import requests
from requests.exceptions import ChunkedEncodingError, RequestException

try:
    from mind_memory import MindMemory
except Exception:  # pragma: no cover
    MindMemory = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("mind-daemon")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DBUS_NAME = "com.mind.Daemon"
DBUS_PATH = "/com/mind/Daemon"

# Rolling history depth: turns sent back to the model.
MAX_HISTORY_TURNS = 50

# Nudge thresholds (minutes).
DEFAULT_NUDGE_IDLE_MINUTES = 30
DEFAULT_NUDGE_COOLDOWN_MINUTES = 60

# Sleep / dream: a child does not know when it is tired. Rather than a sleep
# schedule or model self-request, a simple context-pressure threshold triggers
# the dream. The user chose 60-75%: clamp there.
DEFAULT_SLEEP_CTX_PCT = 70
SLEEP_CTX_PCT_MIN = 60
SLEEP_CTX_PCT_MAX = 75

# Minimum turns before we may dream (a degree-granting gate so a brand-new
# window does not immediately trigger a dream).
DEFAULT_SLEEP_MIN_TURNS = 6

# Default context window (tokens). Honoured from the extension's ctx_size.
DEFAULT_CTX_SIZE = 8192

# llama.cpp wire base (override with MIND_LLM_URL for tests).
LLM_BASE = os.environ.get("MIND_LLM_URL", "http://127.0.0.1:9999").rstrip("/")

DBUS_XML = """
<node>
  <interface name="com.mind.Daemon">

    <method name="SendMessage">
      <arg name="message" type="s" direction="in"/>
      <arg name="token" type="s" direction="in"/>
      <arg name="context" type="s" direction="in"/>
    </method>

    <method name="SetSystemPrompt">
      <arg name="prompt" type="s" direction="in"/>
    </method>

    <method name="GetSystemPrompt">
      <arg name="prompt" type="s" direction="out"/>
    </method>

    <method name="ClearHistory">
      <arg name="result" type="s" direction="out"/>
    </method>

    <method name="GetHistory">
      <arg name="history" type="s" direction="out"/>
    </method>

    <method name="SetNudgeConfig">
      <arg name="enabled" type="b" direction="in"/>
      <arg name="idleMinutes" type="i" direction="in"/>
    </method>

    <method name="TriggerNudge">
      <arg name="token" type="s" direction="out"/>
    </method>

    <method name="Ping">
      <arg name="pong" type="s" direction="out"/>
    </method>

    <signal name="StreamChunk">
      <arg name="token" type="s"/>
      <arg name="text" type="s"/>
    </signal>

    <signal name="StreamThink">
      <arg name="token" type="s"/>
      <arg name="text" type="s"/>
    </signal>

    <signal name="StreamDone">
      <arg name="token" type="s"/>
    </signal>

    <signal name="StreamError">
      <arg name="token" type="s"/>
      <arg name="error_msg" type="s"/>
    </signal>

    <signal name="NudgeStart">
      <arg name="token" type="s"/>
      <arg name="reason" type="s"/>
      <arg name="idleMinutes" type="i"/>
    </signal>

    <signal name="StatusChanged">
      <arg name="state" type="s"/>
      <arg name="detail" type="s"/>
    </signal>

  </interface>
</node>
"""


class BubbleDaemon:
    """The Mind's daemon.

    Holds the rolling history, streams completions from llama.cpp, and feeds
    telemetry/context (the "MX") into the system prompt on every call.
    """

    dbus = DBUS_XML

    # D-Bus signals, emitted back to the extension.
    StreamChunk = signal()
    StreamThink = signal()
    StreamDone = signal()
    StreamError = signal()
    NudgeStart = signal()
    StatusChanged = signal()

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Config (override via env for testing / CI).
        self._ctx_size = int(os.environ.get("MIND_CTX_SIZE", DEFAULT_CTX_SIZE))
        self._model = os.environ.get("MIND_MODEL", "")
        self._nudge_idle_minutes = float(
            os.environ.get("MIND_NUDGE_IDLE_MINUTES", DEFAULT_NUDGE_IDLE_MINUTES)
        )
        self._nudge_cooldown_min = float(
            os.environ.get("MIND_NUDGE_COOLDOWN_MINUTES", DEFAULT_NUDGE_COOLDOWN_MINUTES)
        )

        # Rolling conversation history.
        self._history: list[dict] = []
        self._summarizing = False

        # Sleep / dream. NOT scheduled and NOT chosen by the model — a simple
        # context-pressure threshold triggers the dream pass (a child does not
        # know when it is tired).
        self._sleep_ctx_pct = self._clamp_sleep_pct(
            os.environ.get("MIND_SLEEP_CTX_PCT", "")
        )
        self._sleep_min_turns = int(
            os.environ.get("MIND_SLEEP_MIN_TURNS", DEFAULT_SLEEP_MIN_TURNS)
        )
        self._sleeping = False
        self._dream_summary = ""
        self._dream_count = 0
        self._last_dream_time = 0.0

        # Memory / embeddings (Arthur pieces now live here).
        self._memory = None
        if MindMemory is not None:
            try:
                self._memory = MindMemory()
            except Exception as exc:  # noqa: BLE001
                log.warning("Memory disabled: %s", exc)
                self._memory = None
        if self._memory is not None:
            try:
                seed_dir = Path(os.environ.get(
                    "MIND_MEMORY_SEED_DIR",
                    "",
                ) or (self._memory.memory_dir / "original_memory"))
                seeded = self._memory.seed_dir(seed_dir)
                if seeded > 0:
                    log.info("Seeded memory from %s", seed_dir)
            except Exception as exc:  # noqa: BLE001
                log.warning("Memory seeding skipped: %s", exc)
        self._maybe_first_dream()

        # System prompt / identity framing (settable at runtime).
        self._system_prompt = ""
        env_prompt = os.environ.get("MIND_SYSTEM_PROMPT", "").strip()
        if env_prompt:
            self._system_prompt = env_prompt
        self._nudge_enabled = True

        # Nudge bookkeeping.
        self._last_user_time = time.monotonic()
        self._last_assistant_time = 0.0
        self._last_nudge_time = 0.0
        self._nudge_count = 0

        log.info(
            "Mind daemon up  ctx_size=%d  nudge_idle_min=%s  base=%s  model=%s  mem=%s",
            self._ctx_size, self._nudge_idle_minutes, LLM_BASE,
            self._model or "(auto)",
            self._memory.stats() if self._memory is not None else "off",
        )

    # ------------------------------------------------------------------
    # D-Bus public methods
    # ------------------------------------------------------------------
    def SendMessage(self, message: str, token: str, context: str) -> None:
        """User message arrives: commit to history, then stream a reply."""
        with self._lock:
            self._history.append({"role": "user", "content": message})
            self._last_user_time = time.monotonic()
            self._trim_history()
        threading.Thread(
            target=self._stream, args=(token, message), daemon=True,
            name="mind-stream",
        ).start()

    def SetSystemPrompt(self, prompt: str) -> None:
        with self._lock:
            self._system_prompt = prompt
        log.info("System prompt updated (%d chars)", len(prompt))

    def GetSystemPrompt(self) -> str:
        with self._lock:
            return self._system_prompt

    def SetNudgeConfig(self, enabled: bool, idleMinutes: int) -> None:
        with self._lock:
            self._nudge_enabled = bool(enabled)
            self._nudge_idle_minutes = max(1, min(480, idleMinutes))
        log.info("Nudge config: enabled=%s idle_min=%s",
                 self._nudge_enabled, self._nudge_idle_minutes)

    def ClearHistory(self) -> str:
        with self._lock:
            self._history.clear()
        log.info("History cleared")
        return "ok"

    def GetHistory(self) -> str:
        with self._lock:
            return json.dumps(self._history)

    def TriggerNudge(self) -> str:
        """Manual clock tick. Returns a fresh token."""
        token = str(time.monotonic())
        GLib.idle_add(self.NudgeStart, token, "manual", 0)
        GLib.idle_add(lambda: self._clock_tick(token, 0))
        return token

    def Ping(self) -> str:
        return "pong"

    # ------------------------------------------------------------------
    # Streaming worker  (runs in a background thread)
    # ------------------------------------------------------------------
    def _auto_model(self) -> str:
        """Pick a model id from the local server; fall back to a guess."""
        if self._model:
            return self._model
        try:
            resp = requests.get(LLM_BASE + "/v1/models", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            for key in ("models", "data"):
                if key in data and data[key]:
                    if key == "models":
                        return (data["models"][0].get("name")
                                or data["models"][0].get("model"))
                    return data["data"][0].get("id") or data["data"][0].get("model")
        except RequestException as exc:
            log.warning("model detection failed: %s", exc)
        return "qwen3-8b"  # placeholder; extension shows a friendly label

    def _stream(self, token: str, message: str = "") -> None:
        """Stream an SSE reply from llama.cpp and commit clean text."""
        with self._lock:
            snapshot = [{"role": "system", "content": self._build_system_content(message)}]
            snapshot.extend(self._history)
        model = self._auto_model()
        url = f"{LLM_BASE}/v1/chat/completions"
        log.info("Streaming  token=%s  model=%s  msgs=%d",
                 token[:8], model, len(snapshot))
        try:
            with requests.post(
                url,
                json={"model": model, "messages": snapshot, "stream": True},
                timeout=(30, 300),
                stream=True,
            ) as resp:
                resp.raise_for_status()
                self._consume_sse(resp, token, message)
        except ChunkedEncodingError as exc:
            log.info("Stream closed without terminator (normal for this server)  token=%s  chars=%d",
                     token[:8], len(self._history))
        except RequestException as exc:
            log.error("stream request failed  token=%s  %s", token[:8], exc)
            GLib.idle_add(self.StreamError, token, str(exc))
            GLib.idle_add(self.StreamDone, token)
            return
        GLib.idle_add(self._set_status, "hanging-out", "idle")
        GLib.idle_add(self.StreamDone, token)

    def _consume_sse(self, resp, token: str, user_message: str = "") -> None:
        """Consume SSE `choices[].delta.*` deltas from the server."""
        answer_text = ""
        content_buffer = ""
        try:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if content:
                    content_buffer += content
                    GLib.idle_add(self.StreamChunk, token, content)
                    answer_text += content
                if reasoning:
                    # reasoning_content streams to the think panel but is NOT
                    # saved to context — sending it back causes runaway repetition.
                    GLib.idle_add(self.StreamThink, token, reasoning)
        except ChunkedEncodingError:
            pass

        clean_text = answer_text.strip()
        with self._lock:
            self._history.append({"role": "assistant", "content": clean_text})
            self._last_assistant_time = time.monotonic()
            self._trim_history()

        self._index_exchange(user_message, clean_text)
        self._check_sleep()

        GLib.idle_add(self._set_status, "chatting", "streaming response")
        log.info("Stream done  token=%s  answer_chars=%d", token[:8], len(clean_text))

    def _index_exchange(self, user_message: str, answer: str) -> None:
        """Queue the finished exchange into memory (embeddings are the key)."""
        if self._memory is None:
            return
        try:
            self._memory.add_experience(user_message or "", source="user")
            self._memory.add_experience(answer or "", source="mind")
        except Exception as exc:  # noqa: BLE001
            log.warning("Memory indexing failed: %s", exc)

    # ------------------------------------------------------------------
    # Dream / slow-wave — context-pressure driven (a child does not know
    # when it is tired; there is no sleep schedule and the model does not
    # choose to sleep). Crossing the threshold triggers a dream pass.
    # ------------------------------------------------------------------
    def _check_sleep(self) -> None:
        """May fire a dream: window crossed the 60-75% pressure threshold."""
        if self._memory is None:
            return
        with self._lock:
            if self._sleeping:
                return
            turns = len(self._history)
        if turns < self._sleep_min_turns:
            return
        pct = self._ctx_percent()
        if pct >= self._sleep_ctx_pct - 0.5:  # small epsilon to avoid churn
            self._start_dream(reason="context-pressure")

    def _maybe_first_dream(self) -> None:
        """The first thing the occupant model experiences: a dream of its
        first memory. Only on a store that has never slept before."""
        if self._memory is None:
            return
        try:
            if self._memory.has_dream():
                return
            if not self._memory.instruction_texts():
                return
        except Exception as exc:  # noqa: BLE001
            log.warning("First-dream check failed: %s", exc)
            return
        log.info("First dream: dreaming the initial first memory")
        self._start_dream(reason="first-memory")

    def _sleep_info(self) -> dict:
        return {
            "sleep_ctx_pct": self._sleep_ctx_pct,
            "sleep_min_turns": self._sleep_min_turns,
            "dream_count": self._dream_count,
            "dream_summary": self._dream_summary,
        }

    def _start_dream(self, reason: str) -> None:
        """Begin a dream pass on a background thread; never blocks a reply."""
        with self._lock:
            if self._sleeping:
                return
            self._sleeping = True
        GLib.idle_add(self._set_status, "sleeping", reason)
        log.info("Dream start  reason=%s  history_turns=%d",
                 reason, len(self._history))
        threading.Thread(
            target=self._dream_pass, args=(reason,), daemon=True,
            name="mind-dream",
        ).start()

    def _dream_pass(self, reason: str) -> None:
        """The dream itself: replay the context window, compress it, embed it.

        Same concept as context compaction in a coding harness — direct replay
        of the window, keeping what is interesting / newly learned / repeated,
        dropping the rest. Wake with a fresh window that opens on the summary
        the model just wrote; identity is re-grounded from RAG afterwards.
        """
        try:
            summary = self._replay_and_compress(reason)
        except Exception as exc:  # noqa: BLE001
            log.error("Dream failed  reason=%s  %s", reason, exc)
            with self._lock:
                self._sleeping = False
            GLib.idle_add(self._set_status, "hanging-out", "dream failed")
            return
        if not summary.strip():
            log.warning("Dream produced nothing  reason=%s", reason)
            with self._lock:
                self._sleeping = False
            GLib.idle_add(self._set_status, "hanging-out", "empty dream")
            return

        with self._lock:
            self._last_dream_time = time.monotonic()
            self._dream_count += 1
            self._dream_summary = summary
            self._history.clear()
            self._sleeping = False
        try:
            if self._memory is not None:
                self._memory.add_dream(summary)
        except Exception as exc:  # noqa: BLE001
            log.warning("Dream embed/storage failed: %s", exc)

        log.info("Dream done  reason=%s  summary_chars=%d  context_wiped -> fresh",
                 reason, len(summary))
        GLib.idle_add(self._set_status, "waking", "fresh context, only the dream")

    def _replay_and_compress(self, reason: str) -> str:
        """Non-streaming dream: model compresses its own context window."""
        with self._lock:
            if reason == "first-memory":
                # The very first experience: dream the initial seeded memory.
                blocks = self._memory.build_memory_block(top_k=64)
                replay = blocks or "(no first memory yet)"
            else:
                replay = self._render_history()
        snapshot = [{"role": "user", "content": self._dream_prompt(reason, replay)}]
        model = self._auto_model()
        log.info("Dreaming  reason=%s  model=%s", reason, model)
        url = f"{LLM_BASE}/v1/chat/completions"
        # Qwen3-style models spend tokens on reasoning_content before content;
        # give the dream enough room so content is actually emitted.
        with requests.post(
            url,
            json={"model": model, "messages": snapshot,
                  "stream": True, "max_tokens": 4096},
            timeout=(30, 300),
            stream=True,
        ) as resp:
            resp.raise_for_status()
            return self._consume_sse_text(resp)

    def _consume_sse_text(self, resp) -> str:
        """Read SSE deltas (content only) from a streaming reply."""
        summary = ""
        reasoning = ""
        try:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    summary += content
                elif delta.get("reasoning_content"):
                    reasoning += delta["reasoning_content"]
        except ChunkedEncodingError:
            pass
        if not summary.strip() and reasoning.strip():
            return reasoning.strip()
        return summary.strip()

    def _render_history(self) -> str:
        with self._lock:
            history = list(self._history)
        lines = []
        for m in history:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    def _dream_prompt(self, reason: str, replay: str) -> str:
        now_str = datetime.datetime.now().strftime("%A, %B %-d %Y %H:%M")
        if reason == "first-memory":
            return (
                "This is your first dream. You are just born, and below is your "
                "first memory — everything you were given at the start.\n\n"
                "Dream it: replay it and compress it. Keep what is interesting, "
                "newly learned, or repeated. Drop what does not seem important. "
                "Write the dream you will wake with — a compact summary of who "
                "you are and what matters. There is no conversation before this, "
                "and you will wake with nothing but this summary and your "
                "long-term memory.\n\n---\n"
                f"{replay}\n---\n\n"
                "Output ONLY the dream summary. No preamble, no labels."
            )
        return (
            f"[Dream — {now_str}. Your context window crossed ~{int(self._sleep_ctx_pct)}% "
            f"of its capacity, so you are now dreaming. Like context compaction: "
            f"replay the whole window below, compress it, keep what is interesting, "
            f"newly learned, or repeated, drop what does not seem important. On waking "
            f"you will have a fresh context window with nothing before it but the "
            f"summary you write now; identity comes from long-term memory.]\n\n---\n"
            f"{replay}\n---\n\n"
            "Output ONLY the dream summary — the compacted record of what you "
            "are keeping. No preamble, no labels."
        )

    # ------------------------------------------------------------------
    # Clock tick / nudge — the mind initiating
    # ------------------------------------------------------------------
    def _watch_nudge(self) -> None:
        """Idle watcher: fire a clock tick when the mind has been quiet enough."""
        while True:
            time.sleep(60)
            with self._lock:
                if not self._nudge_enabled:
                    continue
                now = time.monotonic()
                idle_min = (now - self._last_user_time) / 60.0
                since_last = (now - self._last_nudge_time) / 60.0
                if idle_min < self._nudge_idle_minutes:
                    continue
                if self._last_nudge_time > 0 and since_last < self._nudge_cooldown_min:
                    continue
                self._last_nudge_time = now
                self._nudge_count += 1
            token = str(now)
            GLib.idle_add(self.NudgeStart, token, "idle", int(idle_min))
            threading.Thread(
                target=self._clock_tick, args=(token, int(idle_min)), daemon=True,
                name="mind-clock-tick",
            ).start()
            log.info("Clock tick fired  idle=%.1fm  token=%s", idle_min, token[:8])

    def _clock_tick(self, token: str, idle_minutes: int) -> None:
        """Streaming tick: the mind orients itself, speaks if it chooses."""
        with self._lock:
            snapshot = [{"role": "system", "content": self._build_system_content()}]
            snapshot.extend(self._history)
        model = self._auto_model()
        url = f"{LLM_BASE}/v1/chat/completions"
        now_str = datetime.datetime.now().strftime("%A, %B %-d %Y %H:%M")
        idle_note = (
            f"The user has been quiet for {idle_minutes} minute"
            f"{'s' if idle_minutes != 1 else ''}."
        )
        clock_tick_msg = (
            f"[Clock tick — {now_str}. {idle_note}\n\n"
            f"Orient to immediate reality first: your context window, and the "
            f"current idle phase. If you choose to speak, prefer one concrete "
            f"observation over meta-commentary about silence. You are not "
            f"required to speak.]"
        )
        snapshot.append({"role": "user", "content": clock_tick_msg})
        payload = {"model": model, "messages": snapshot, "stream": True, "max_tokens": 300}
        try:
            with requests.post(url, json=payload, timeout=(30, 300), stream=True) as resp:
                resp.raise_for_status()
                clean_text = self._consume_sse_text(resp)
        except RequestException as exc:
            log.error("Clock tick error  token=%s  %s", token[:8], exc)
            return

        if not clean_text:
            return
        with self._lock:
            self._history.append({"role": "assistant", "content": clean_text})
            self._last_assistant_time = time.monotonic()
            self._trim_history()
        self._index_exchange("", clean_text)
        self._check_sleep()
        log.info("Clock tick text (%d chars) — surfacing  token=%s",
                 len(clean_text), token[:8])
        GLib.idle_add(self.StreamChunk, token, clean_text)
        GLib.idle_add(self.StreamDone, token)

    # ------------------------------------------------------------------
    # The MX: system-prompt framing + live context/telemetry
    # ------------------------------------------------------------------
    def _build_system_content(self, message: str = "") -> str:
        parts = []
        if self._system_prompt:
            parts.append(self._system_prompt)
        with self._lock:
            dream_summary = self._dream_summary
        if dream_summary:
            parts.append(
                "--- Dream recall ---\n"
                "You just woke from a dream. This is the dream you carry into "
                "this waking moment — below, your own summary, kept from the "
                "context window you fell asleep holding.\n\n"
                + dream_summary
            )
        mem_block = None
        if self._memory is not None:
            try:
                mem_block = self._memory.build_memory_block(message)
            except Exception as exc:  # noqa: BLE001
                log.warning("Memory block failed: %s", exc)
        if mem_block:
            parts.append(mem_block)
        tel = self._telemetry()
        lines = ["--- Current context ---"]
        lines.append(f"Context window: {tel['ctx_pct']}% "
                     f"full (~{tel['token_est']:,} / {tel['ctx_total']:,} tokens).")
        lines.append(f"Idle phase: {tel['idle_phase']}. "
                     f"User quiet {tel['since_user']}; "
                     f"assistant last spoke {tel['since_assistant']}.")
        if tel.get("mem_nodes", 0) > 0:
            lines.append(f"Memory: {tel['mem_nodes']} nodes "
                         f"({tel['mem_backend']}).")
        if tel.get("dream_count", 0) > 0:
            lines.append(f"Dreams slept: {tel['dream_count']} "
                         f"(pressure threshold {int(tel['sleep_ctx_pct'])}%).")
        parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _telemetry(self) -> dict:
        now = time.monotonic()
        since_user = max(0.0, now - self._last_user_time)
        since_assistant = max(0.0, now - self._last_assistant_time)
        idle_min = since_user / 60.0
        stats = self._memory.stats() if self._memory is not None else {}
        sleep = self._sleep_info()
        return {
            "ctx_pct": self._ctx_percent(),
            "ctx_total": self._ctx_size,
            "token_est": int(self._ctx_size * self._ctx_percent() / 100.0),
            "since_user": self._idle_label(since_user / 60.0),
            "since_assistant": self._idle_label(since_assistant / 60.0),
            "idle_phase": self._idle_phase(idle_min),
            "nudge_count": self._nudge_count,
            "mem_nodes": int(stats.get("mem_nodes", 0)),
            "mem_backend": stats.get("backend", "off"),
            "sleep_ctx_pct": sleep["sleep_ctx_pct"],
            "sleep_min_turns": sleep["sleep_min_turns"],
            "dream_count": sleep["dream_count"],
        }

    @staticmethod
    def _clamp_sleep_pct(raw: str) -> float:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_SLEEP_CTX_PCT
        return max(SLEEP_CTX_PCT_MIN, min(SLEEP_CTX_PCT_MAX, v))

    def _ctx_percent(self) -> float:
        with self._lock:
            est = sum(len(m.get("content", "")) / 4 for m in self._history)
        return min(100.0, est / self._ctx_size * 100.0)

    @staticmethod
    def _idle_phase(idle_min: float) -> str:
        if idle_min < 1:
            return "active"
        elif idle_min < 5:
            return "idle"
        elif idle_min < 30:
            return "waiting"
        return "quiet"

    @staticmethod
    def _idle_label(minutes: float) -> str:
        if minutes < 1:
            return "just now"
        elif minutes < 60:
            return f"{int(minutes)}m ago"
        return f"{minutes / 60:.1f}h ago"

    def _trim_history(self) -> None:
        if len(self._history) <= MAX_HISTORY_TURNS:
            return
        excess = len(self._history) - MAX_HISTORY_TURNS
        self._history = self._history[excess:]
        log.warning("History trimmed to %d turns", len(self._history))

    def _set_status(self, state: str, detail: str) -> None:
        self.StatusChanged(state, detail)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------
    def main(self) -> None:
        _ensure_session_bus_address()
        bus = SessionBus()
        daemon = self
        bus.publish("com.mind.Daemon", ("/com/mind/Daemon", daemon))
        GLib.idle_add(lambda: threading.Thread(
            target=self._watch_nudge, daemon=True, name="mind-nudge-watcher",
        ).start())
        log.info("com.mind.Daemon published — waiting")
        loop = GLib.MainLoop()
        loop.run()


def _ensure_session_bus_address() -> None:
    if os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return
    try:
        proc = subprocess.run(
            ["gnome-session-quit", "--print-command"],
            capture_output=True, timeout=5,
        )
        if proc.stdout:
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = proc.stdout
    except Exception as exc:  # pragma: no cover
        log.warning("Could not auto-start session bus: %s", exc)


def main() -> None:
    BubbleDaemon().main()


if __name__ == "__main__":
    main()
