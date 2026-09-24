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

Memory / RAG / embeddings, scratch working-notes, the self-prompt revision
protocol, summarization, dream/nap and the slow-wave night cycle are
intentionally OMITTED — they are Arthur pieces and come later.

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
        self._lock = threading.Lock()

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
        GLib.idle_add(lambda: self._stream(token, message))

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
            GLib.idle_add(lambda: self._clock_tick(token, int(idle_min)))
            log.info("Clock tick fired  idle=%.1fm  token=%s", idle_min, token[:8])

    def _clock_tick(self, token: str, idle_minutes: int) -> None:
        """Non-streaming tick: the mind orients itself, speaks if it chooses."""
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
        payload = {"model": model, "messages": snapshot, "stream": False, "max_tokens": 300}
        try:
            resp = requests.post(url, json=payload, timeout=(10, 90))
            resp.raise_for_status()
            clean_text = (
                resp.json().get("choices", [{}])[0]
                     .get("message", {}).get("content", "").strip()
            )
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
        parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _telemetry(self) -> dict:
        now = time.monotonic()
        since_user = max(0.0, now - self._last_user_time)
        since_assistant = max(0.0, now - self._last_assistant_time)
        idle_min = since_user / 60.0
        stats = self._memory.stats() if self._memory is not None else {}
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
        }

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
