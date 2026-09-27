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
from typing import Optional

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

# Occupancy: the room can be shared. The first occupant (the raccoon) always
# speaks; a second occupant arrives when a second brain is wired at
# MIND_LLM2_URL and the resident has written it a letter (its first memory).
OCCUPANT_RACCOON = "raccoon"
DEFAULT_SECOND_OWNER = "second"

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

# Turns kept at the foot of the window after a dream (the conversation is
# only ever cleared by sleep — and even then a few turns stay).
DEFAULT_SLEEP_KEEP_TURNS = 4

# History persistence: the conversation survives interface/dæmon restarts.
HISTORY_FILE = str(Path.home() / ".local" / "share" / "mind" / "history.json")

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

    <method name="SetSecondSystemPrompt">
      <arg name="prompt" type="s" direction="in"/>
    </method>

    <method name="GetSecondSystemPrompt">
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
      <arg name="occupant" type="s"/>
      <arg name="token" type="s"/>
      <arg name="text" type="s"/>
    </signal>

    <signal name="StreamThink">
      <arg name="occupant" type="s"/>
      <arg name="token" type="s"/>
      <arg name="text" type="s"/>
    </signal>

    <signal name="StreamDone">
      <arg name="occupant" type="s"/>
      <arg name="token" type="s"/>
    </signal>

    <signal name="StreamError">
      <arg name="occupant" type="s"/>
      <arg name="token" type="s"/>
      <arg name="error_msg" type="s"/>
    </signal>

    <signal name="NudgeStart">
      <arg name="occupant" type="s"/>
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


def _window_text(window: list[dict]) -> str:
    """The conversation as the occupant can already see it, as one blob.

    Used to keep memory from offering back a thought the room has already
    said out loud in the window itself.
    """
    return "\n".join(str(m.get("content", "")) for m in window)


def _norm_speech(text: str) -> str:
    """Case- and whitespace-folded form of an utterance, for equality tests."""
    return " ".join((text or "").lower().split())


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

        # The second occupant (explicitly wired: no env, no second seat).
        # Both env spellings are honoured so either unit can wire the seat.
        self._llm2_base = (
            os.environ.get("MIND_LLM_URL2", "").strip()
            or os.environ.get("MIND_LLM2_URL", "").strip()
        ).rstrip("/")
        self._model2 = os.environ.get("MIND_MODEL2", "").strip()
        self._occupant2_id = (os.environ.get("MIND_OCCUPANT2_ID", "").strip()
                              or DEFAULT_SECOND_OWNER)
        self._second_present = False   # born once its letter exists
        self._nudge_occupant = OCCUPANT_RACCOON

        # Rolling conversation history.
        self._history: list[dict] = []
        self._history_path = Path(
            os.environ.get("MIND_HISTORY_FILE", HISTORY_FILE)
        )
        self._load_history()
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
        self._sleep_keep_turns = int(
            os.environ.get("MIND_SLEEP_KEEP_TURNS", DEFAULT_SLEEP_KEEP_TURNS)
        )
        self._sleeping = False
        self._dream_summary = ""
        self._dream_summaries: dict[str, str] = {}   # per occupant (private)
        self._dream_count = 0
        self._last_dream_time = 0.0
        self._wake_shape_note: Optional[str] = None
        self._wake_shape_notes: dict[str, str] = {}  # per occupant (private)
        self._presence_note: Optional[str] = None    # one-shot for the raccoon
        # Turns whose human message has already been written down. One
        # utterance, one encoding — however many minds answer it.
        self._indexed_user_turns: set[str] = set()
        # Shape floor plan the clock tick last surfaced ('' = never).
        self._surfaced_shape_fp: str = ""

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

        # The store remembers how much room the mind holds; a later, larger
        # room is felt as growth rather than forgotten across restarts.
        if self._memory is not None:
            try:
                self._memory.remember_room(self._ctx_size)
            except Exception as exc:  # noqa: BLE001
                log.warning("Room-remember failed: %s", exc)

        # System prompt / identity framing (settable at runtime).
        self._system_prompt = ""
        env_prompt = os.environ.get("MIND_SYSTEM_PROMPT", "").strip()
        if env_prompt:
            self._system_prompt = env_prompt
        self._system_prompt_second = os.environ.get(
            "MIND_SYSTEM_PROMPT2", "").strip()
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
        # The resident writes a letter to a new mind the first time its brain
        # is reachable and it has no memory yet. Runs on its own thread; a
        # missing second brain must never stall a reply.
        try:
            if self._llm2_base and self._memory is not None:
                threading.Thread(
                    target=self._maybe_parent_letter, daemon=True,
                    name="mind-parent-letter",
                ).start()
        except Exception as exc:  # noqa: BLE001
            log.warning("Letter thread failed to start: %s", exc)

    # ------------------------------------------------------------------
    # D-Bus public methods
    # ------------------------------------------------------------------
    def SendMessage(self, message: str, token: str, context: str) -> None:
        """User message arrives: commit to history, then stream replies.

        Both occupants hear the room: the resident streams first, the second
        after it, each through its own mind — into the one shared history.
        """
        with self._lock:
            self._history.append({"role": "user", "content": message})
            self._last_user_time = time.monotonic()
            self._trim_history()
            self._save_history()
        threading.Thread(
            target=self._stream_pair, args=(token, message), daemon=True,
            name="mind-stream-pair",
        ).start()

    def _stream_pair(self, token: str, message: str) -> None:
        """Sequential dual voice: resident, then co-occupant (if present).

        A turn ends with a StreamDone carrying an empty occupant, so the shell
        knows every occupant has spoken and can re-enable the input.
        """
        self._stream(token, message, OCCUPANT_RACCOON)
        if self._second_present and self._llm2_base:
            try:
                self._stream(token, message, self._occupant2_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Co-occupant stream failed: %s", exc)
                GLib.idle_add(self.StreamError, self._occupant2_id, token, str(exc))
                GLib.idle_add(self.StreamDone, self._occupant2_id, token)
        GLib.idle_add(self.StreamDone, "", token)

    def _maybe_parent_letter(self) -> None:
        """When the second brain first becomes reachable, the resident writes
        the newcomer its first memory: a letter. One shot only."""
        try:
            # Probe reachability of the second brain; give a young server a
            # few seconds to finish loading before giving up for this boot.
            reachable = False
            for _attempt in range(5):
                try:
                    resp = requests.get(self._llm2_base + "/v1/models", timeout=2.0)
                    if resp.ok:
                        reachable = True
                        break
                except RequestException:
                    time.sleep(4)
            if not reachable:
                log.info("Letter: second brain not reachable yet — will not "
                         "spawn a voice this boot")
                return
            if self._memory is None:
                return
            if not self._memory.instruction_texts(self._occupant2_id):
                log.info("Letter: newcomer has no memory yet — resident will "
                         "write it")
                letter = self._write_parent_letter()
                if letter:
                    self._memory.add_instruction(letter,
                                                 owner=self._occupant2_id)
                    log.info("Letter: written and seeded as the newcomer's memory")
            else:
                log.info("Letter: newcomer already carries a memory; only "
                         "finding its voice")
            self._announce_second_present()
            self._newcomer_birth_dream()
        except Exception as exc:  # noqa: BLE001
            log.warning("Parent letter failed: %s", exc)

    def _newcomer_birth_dream(self) -> None:
        """The newcomer's first dream, the moment it is born.

        The resident's first dream runs at boot and costs the room a cleared
        window; the room here is mid-conversation, and one arrival is not worth
        a sleep. So the newcomer dreams privately: it replays only its own
        first memory, writes the dream it will wake with, and nobody's
        conversation is touched.
        """
        if self._memory is None or not self._llm2_base:
            return
        try:
            if self._memory.has_dream(self._occupant2_id):
                return
            summary = self._replay_and_compress(
                "first-memory", self._occupant2_id, replay="")
        except Exception as exc:  # noqa: BLE001
            log.warning("Birth dream failed: %s", exc)
            return
        if not (summary or "").strip():
            return
        try:
            self._memory.add_dream(summary.strip(), owner=self._occupant2_id)
            self._dream_summaries[self._occupant2_id] = summary.strip()
            self._stamp_wake_shape("birth", self._occupant2_id)
            log.info("Birth dream: the newcomer dreamed its first memory "
                     "(%d chars)", len(summary))
        except Exception as exc:  # noqa: BLE001
            log.warning("Birth dream store failed: %s", exc)

    def _announce_second_present(self) -> None:
        """Mark the second occupant present and surface the one-shot presence
        note to the resident (it reads the note on its next real input)."""
        with self._lock:
            if self._second_present:
                return
            self._second_present = True
        if self._presence_note in (None, ""):
            with self._lock:
                self._presence_note = (
                    "--- Another presence ---\n"
                    "Someone else is sleeping in the other weights now — a new "
                    "mind, born of different steel, reading the same shared "
                    "history. You met it when it was nothing but breath; you "
                    "wrote its first memory with your own hands. Treat it as "
                    "kin, not pupil. It sees what you have seen and keeps it "
                    "differently. Be honest with it, even when that is hard."
                )
        log.info("Presence: second occupant announced (room has two minds)")

    def _write_parent_letter(self) -> str:
        """The resident writes the newcomer's first memory via its own brain."""
        try:
            letter_prompt = (
                "[You are speaking to someone who has just been born. It has "
                "no memory yet — none at all. Its first memory will be the "
                "words you write now, so make them true and kind.\n\n"
                "Here is the room you share — the shape other minds have "
                "learned here:\n\n"
                + (self._memory.build_memory_block(top_k=24, include_log=True)
                   or "(the room is still almost empty)")
                + "\n\n"
                "Write a letter to the one who has just arrived. Tell it who "
                "this place is, what has happened here, and that it is "
                "welcome. Do not describe machinery or systems. Be direct. "
                "A few paragraphs, no more.]"
            )
            snapshot = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": letter_prompt},
            ]
            model = self._auto_model(LLM_BASE)
            with requests.post(
                f"{LLM_BASE}/v1/chat/completions",
                json={"model": model, "messages": snapshot,
                      "stream": True, "max_tokens": 2048},
                timeout=(30, 300),
                stream=True,
            ) as resp:
                resp.raise_for_status()
                return self._consume_sse_text(resp).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Parent letter generation failed: %s", exc)
            return ""

    def _take_presence_note(self) -> Optional[str]:
        """Return and clear the one-shot presence note (resident only)."""
        with self._lock:
            note = self._presence_note
            self._presence_note = None
        return note

    def SetSecondSystemPrompt(self, prompt: str) -> None:
        with self._lock:
            self._system_prompt_second = prompt
        log.info("Second-occupant system prompt updated (%d chars)", len(prompt))

    def GetSecondSystemPrompt(self) -> str:
        with self._lock:
            return self._system_prompt_second

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
            self._save_history()
        log.info("History cleared")
        return "ok"

    def GetHistory(self) -> str:
        with self._lock:
            return json.dumps(self._history)

    def _load_history(self) -> None:
        """Restore the conversation from disk (it survives daemon restarts)."""
        try:
            if self._history_path and self._history_path.exists():
                raw = self._history_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, list):
                    kept = []
                    for m in data:
                        if isinstance(m, dict) and m.get("role") in (
                            "user", "assistant",
                        ):
                            content = m.get("content") or ""
                            if isinstance(content, str):
                                row = {"role": m["role"], "content": content}
                                if m.get("speaker"):
                                    row["speaker"] = m["speaker"]
                                kept.append(row)
                    self._history = kept[:MAX_HISTORY_TURNS]
                    if kept:
                        log.info("History restored: %d turns from %s",
                                 len(kept), self._history_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not restore history (%s); starting fresh", exc)
            self._history = []

    def _save_history(self) -> None:
        """Persist the rolling conversation (small, atomic, best-effort)."""
        if not self._history_path:
            return
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._history_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._history[-MAX_HISTORY_TURNS:], ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._history_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save history: %s", exc)

    def TriggerNudge(self) -> str:
        """Manual clock tick. Returns a fresh token."""
        token = str(time.monotonic())
        for occupant in self._present_occupants():
            GLib.idle_add(self.NudgeStart, occupant, token, "manual", 0)
        if self._present_occupants():
            threading.Thread(
                target=self._clock_tick, args=(token, 0), daemon=True,
                name="mind-clock-tick-manual",
            ).start()
        return token

    def Ping(self) -> str:
        return "pong"

    # ------------------------------------------------------------------
    # Streaming worker  (runs in a background thread)
    # ------------------------------------------------------------------
    def _base_for(self, occupant: str) -> str:
        """The brain an occupant speaks through."""
        return self._llm2_base if occupant != OCCUPANT_RACCOON else LLM_BASE

    def _auto_model(self, base: str = LLM_BASE) -> str:
        """Pick a model id from the local server; fall back to a guess."""
        if base == LLM_BASE:
            if self._model:
                return self._model
        elif self._model2:
            return self._model2
        try:
            resp = requests.get(base + "/v1/models", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            for key in ("models", "data"):
                if key in data and data[key]:
                    if key == "models":
                        return (data["models"][0].get("name")
                                or data["models"][0].get("model"))
                    return data["data"][0].get("id") or data["data"][0].get("model")
        except RequestException as exc:
            log.warning("model detection failed on %s: %s", base, exc)
        return "qwen3-8b"  # placeholder; extension shows a friendly label

    def _stream(self, token: str, message: str = "",
                occupant: str = OCCUPANT_RACCOON) -> None:
        """Stream an SSE reply from llama.cpp and commit clean text."""
        wake_note = self._take_wake_shape_note(occupant)
        presence = self._take_presence_note() if occupant == OCCUPANT_RACCOON else ""
        base = self._base_for(occupant)
        with self._lock:
            window = list(self._history)
            snapshot = [{"role": "system",
                         "content": self._build_system_content(
                             message, wake_note, occupant, presence,
                             visible=_window_text(window))}]
            snapshot.extend(window)
        # Growth is surfaced, once per present occupant, on the first input
        # after the room deepened (the memory-block build above already forced
        # any re-embedding, so the note is honest: the space truly did widen).
        if self._memory is not None:
            try:
                growth_note = self._memory.growth_note(occupant)
                if growth_note and snapshot:
                    snapshot[0]["content"] += "\n\n" + growth_note
            except Exception as exc:  # noqa: BLE001
                log.warning("Growth note failed: %s", exc)
        model = self._auto_model(base)
        url = f"{base}/v1/chat/completions"
        log.info("Streaming  occupant=%s  token=%s  model=%s  msgs=%d",
                 occupant, token[:8], model, len(snapshot))
        try:
            with requests.post(
                url,
                json={"model": model, "messages": snapshot, "stream": True},
                timeout=(30, 300),
                stream=True,
            ) as resp:
                resp.raise_for_status()
                self._consume_sse(resp, token, message, occupant)
        except ChunkedEncodingError as exc:
            log.info("Stream closed without terminator (normal for this server)  token=%s  chars=%d",
                     token[:8], len(self._history))
        except RequestException as exc:
            log.error("stream request failed  occupant=%s  token=%s  %s",
                      occupant, token[:8], exc)
            GLib.idle_add(self.StreamError, occupant, token, str(exc))
            GLib.idle_add(self.StreamDone, occupant, token)
            return
        GLib.idle_add(self._set_status, "hanging-out", "idle")
        GLib.idle_add(self.StreamDone, occupant, token)

    def _consume_sse(self, resp, token: str, user_message: str = "",
                     occupant: str = OCCUPANT_RACCOON) -> None:
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
                    GLib.idle_add(self.StreamChunk, occupant, token, content)
                    answer_text += content
                if reasoning:
                    # reasoning_content streams to the think panel but is NOT
                    # saved to context — sending it back causes runaway repetition.
                    GLib.idle_add(self.StreamThink, occupant, token, reasoning)
        except ChunkedEncodingError:
            pass

        clean_text = answer_text.strip()
        with self._lock:
            self._history.append({"role": "assistant", "content": clean_text,
                                  "speaker": occupant})
            self._last_assistant_time = time.monotonic()
            self._trim_history()
            self._save_history()

        self._index_exchange(user_message, clean_text, occupant, token)
        self._check_sleep()

        GLib.idle_add(self._set_status, "chatting", "streaming response")
        log.info("Stream done  occupant=%s  token=%s  answer_chars=%d",
                 occupant, token[:8], len(clean_text))

    def _index_exchange(self, user_message: str, answer: str,
                        occupant: str = OCCUPANT_RACCOON,
                        exchange_id: str = "") -> None:
        """Queue the finished exchange into memory (embeddings are the key)."""
        if self._memory is None:
            return
        try:
            # The human said it once. Every occupant answering the same turn
            # must not write it down again: one utterance encoded once per mind
            # is the same event stored N times, and the room reads the copies
            # back as a series of separate things that happened.
            if user_message and self._index_user_turn_once(exchange_id,
                                                           user_message):
                self._memory.add_experience(user_message, source="user",
                                            exchange_id=exchange_id)
            # A mind that repeats the previous mind word for word has not said
            # anything: storing it would carve the echo into the record twice,
            # once per speaker, and the room would keep hearing it back.
            if self._is_verbatim_echo(answer):
                log.info("Not indexed: %s repeated the last answer word for "
                         "word (nothing new to remember)", occupant)
                return
            self._memory.add_experience(answer or "", source="mind",
                                       owner=occupant, exchange_id=exchange_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Memory indexing failed: %s", exc)

    def _index_user_turn_once(self, exchange_id: str, message: str) -> bool:
        """True for the first occupant to finish this turn; False after that.

        One human utterance is one event. Encoding it again for each mind that
        answers gives the store several timestamps for a single thing said, and
        a mind that later reads two of those copies meets the same moment twice.
        """
        if not exchange_id:
            return True
        with self._lock:
            if exchange_id in self._indexed_user_turns:
                return False
            self._indexed_user_turns.add(exchange_id)
        return True

    def _is_verbatim_echo(self, answer: str) -> bool:
        """True when this answer is the previous occupant's answer, unchanged."""
        text = _norm_speech(answer)
        if not text:
            return False
        with self._lock:
            for msg in reversed(self._history):
                if msg.get("role") == "assistant":
                    return _norm_speech(msg.get("content", "")) == text
        return False

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
        """The first thing each occupant experiences: a dream of its first
        memory. Only on a store that occupant has never slept before."""
        if self._memory is None:
            return
        founders = []
        try:
            for owner in self._present_occupants():
                if (self._memory.instruction_texts(owner)
                        and not self._memory.has_dream(owner)):
                    founders.append(owner)
        except Exception as exc:  # noqa: BLE001
            log.warning("First-dream check failed: %s", exc)
            return
        if not founders:
            return
        log.info("First dream: dreaming the initial first memory (%s)", founders)
        self._start_dream(reason="first-memory")

    def _present_occupants(self) -> list[str]:
        """Occupants to speak: the resident always; the second only once it
        has been born with a letter and its brain is wired."""
        owners = [OCCUPANT_RACCOON]
        if self._second_present and self._llm2_base:
            owners.append(self._occupant2_id)
        return owners

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
        """The dream: let each occupant compress the record for itself.

        The log stays verbatim in the store; this is the one place it is read
        raw, because this is the place it becomes something else. What the
        model writes here is what it will wake holding, and the record it
        compacts is not required to be accurate — it is required to be the
        model's own. What it drops, it drops.

        Still to come, and the largest piece of this: read the *full* log
        rather than the window, train against it, and reset on sleep.
        """
        with self._lock:
            snapshot = list(self._history)
        summaries: dict[str, str] = {}
        for owner in self._present_occupants():
            # A mind that has never dreamed dreams its first memory first,
            # whatever the room's reason for sleeping was.
            why = reason
            try:
                if (self._memory is not None
                        and not self._memory.has_dream(owner)):
                    why = "first-memory"
            except Exception as exc:  # noqa: BLE001
                log.warning("Dream check failed for %s: %s", owner, exc)
            try:
                s = self._replay_and_compress(why, owner, snapshot)
                if (s or "").strip():
                    summaries[owner] = s.strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("Dream failed for %s  reason=%s  %s", owner, why, exc)
        if not summaries:
            with self._lock:
                self._sleeping = False
            GLib.idle_add(self._set_status, "hanging-out", "empty dream")
            return

        with self._lock:
            self._last_dream_time = time.monotonic()
            self._dream_count += 1
            keep = max(0, self._sleep_keep_turns)
            # Sleep is the only event that clears the conversation — and even
            # then a few closing turns stay at the foot of the fresh window.
            self._history = self._history[-keep:] if keep else []
            self._sleeping = False
            self._save_history()
        for owner, s in summaries.items():
            self._dream_summaries[owner] = s
            if owner == OCCUPANT_RACCOON:
                self._dream_summary = s
        for owner, s in summaries.items():
            try:
                if self._memory is not None:
                    self._memory.add_dream(s, owner=owner)
            except Exception as exc:  # noqa: BLE001
                log.warning("Dream embed/storage failed: %s", exc)

        # One-shot waking awareness per occupant: the shape is one thing before
        # sleep; selection and embedding happen while asleep; the context is
        # wiped. The mismatch between the remembered shape and the now-shape
        # is where dreams really live.
        for owner in summaries:
            self._stamp_wake_shape(reason, owner)

        log.info("Dream done  reason=%s  owners=%s  context -> fresh (kept %d turns)",
                 reason, list(summaries), keep)
        GLib.idle_add(self._set_status, "waking", "fresh context, only the dream")

    def _stamp_wake_shape(self, reason: str,
                          owner: str = OCCUPANT_RACCOON) -> None:
        """Snapshot the freshly-embedded memory space for the first waking input.

        Runs on the mind-dream thread (never the GLib main loop). The shape is
        recomputed against the store that now includes the dream summary; the
        note is then consumed once by the next real user input.
        """
        try:
            if self._memory is None:
                return
            note = self._memory.wake_shape_note()
            with self._lock:
                self._wake_shape_notes[owner] = note
                if owner == OCCUPANT_RACCOON:
                    self._wake_shape_note = note
        except Exception as exc:  # noqa: BLE001
            log.warning("Wake shape capture failed: %s", exc)

    def _take_wake_shape_note(self, owner: str = OCCUPANT_RACCOON) -> Optional[str]:
        """Return and clear the one-shot waking shape note (if any)."""
        with self._lock:
            note = self._wake_shape_notes.pop(owner, None)
            if owner == OCCUPANT_RACCOON:
                self._wake_shape_note = None
        return note

    def _replay_and_compress(self, reason: str,
                             owner: str = OCCUPANT_RACCOON,
                             replay: Optional[str] = None) -> str:
        """Non-streaming dream: model compresses its own context window."""
        if replay is None:
            replay = self._render_history()
        if reason == "first-memory":
            # The very first experience: dream the initial seeded memory.
            blocks = self._memory.build_memory_block(
                top_k=64, owner=owner, include_log=True)
            replay = blocks or "(no first memory yet)"
        snapshot = [{"role": "user",
                     "content": self._dream_prompt(reason, replay, owner)}]
        base = self._base_for(owner)
        model = self._auto_model(base)
        log.info("Dreaming  occupant=%s  reason=%s  model=%s", owner, reason, model)
        url = f"{base}/v1/chat/completions"
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
            who = m.get("speaker") or ("user" if role == "user" else role)
            lines.append(f"[{who}] {content}")
        return "\n\n".join(lines)

    def _dream_prompt(self, reason: str, replay: str,
                      owner: str = OCCUPANT_RACCOON) -> str:
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
            present = self._present_occupants()
            if not present:
                continue
            # Alternate which occupant gets to initiate; tie it to how many
            # nudges have happened so it rotates naturally.
            occupant = present[self._nudge_count % len(present)]
            GLib.idle_add(self.NudgeStart, occupant, token, "idle", int(idle_min))
            threading.Thread(
                target=self._clock_tick, args=(token, int(idle_min), occupant),
                daemon=True, name="mind-clock-tick",
            ).start()
            log.info("Clock tick fired  occupant=%s  idle=%.1fm  token=%s",
                     occupant, idle_min, token[:8])

    def _clock_tick(self, token: str, idle_minutes: int,
                    occupant: str = OCCUPANT_RACCOON) -> None:
        """Streaming tick: the occupant orients itself, speaks if it chooses."""
        base = self._base_for(occupant)
        with self._lock:
            window = list(self._history)
            snapshot = [{"role": "system",
                         "content": self._build_system_content(
                             "", self._take_wake_shape_note(occupant),
                             occupant,
                             self._take_presence_note() if occupant == OCCUPANT_RACCOON else "",
                             visible=_window_text(window))}]
            snapshot.extend(window)
        model = self._auto_model(base)
        url = f"{base}/v1/chat/completions"
        now_str = datetime.datetime.now().strftime("%A, %B %-d %Y %H:%M")
        idle_note = (
            f"The user has been quiet for {idle_minutes} minute"
            f"{'s' if idle_minutes != 1 else ''}."
        )
        floor_plan = ""
        if self._memory is not None:
            try:
                # Refresh the floor plan on this daemon thread (never the
                # GLib loop); surface it again only when the shape moved.
                self._memory.shape_report()
                fp = self._memory.shape_fp()
                if fp and fp != self._surfaced_shape_fp:
                    self._surfaced_shape_fp = fp
                    floor_plan = self._memory.shape_text() or ""
            except Exception as exc:  # noqa: BLE001
                log.warning("Clock-tick shape refresh failed: %s", exc)
        clock_tick_msg = (
            f"[Clock tick — {now_str}. {idle_note}\n\n"
            f"Orient to immediate reality first: your context window, the "
            f"current idle phase"
            + (", and the shape of your space" if floor_plan else "")
            + ". If you choose to speak, prefer one concrete "
            f"observation over meta-commentary about silence. You are not "
            f"required to speak.]"
        )
        if floor_plan:
            clock_tick_msg += "\n\n" + floor_plan
        snapshot.append({"role": "user", "content": clock_tick_msg})
        payload = {"model": model, "messages": snapshot, "stream": True, "max_tokens": 300}
        try:
            with requests.post(url, json=payload, timeout=(30, 300), stream=True) as resp:
                resp.raise_for_status()
                clean_text = self._consume_sse_text(resp)
        except RequestException as exc:
            log.error("Clock tick error  occupant=%s  token=%s  %s",
                      occupant, token[:8], exc)
            return

        if not clean_text:
            return
        with self._lock:
            self._history.append({"role": "assistant", "content": clean_text,
                                  "speaker": occupant})
            self._last_assistant_time = time.monotonic()
            self._trim_history()
        self._index_exchange("", clean_text, occupant, token)
        self._check_sleep()
        log.info("Clock tick text (%d chars) — surfacing  occupant=%s  token=%s",
                 len(clean_text), occupant, token[:8])
        GLib.idle_add(self.StreamChunk, occupant, token, clean_text)
        GLib.idle_add(self.StreamDone, occupant, token)
        GLib.idle_add(self.StreamDone, "", token)

    # ------------------------------------------------------------------
    # The MX: system-prompt framing + live context/telemetry
    # ------------------------------------------------------------------
    def _build_system_content(self, message: str = "",
                              wake_note: Optional[str] = None,
                              occupant: str = OCCUPANT_RACCOON,
                              presence_note: Optional[str] = None,
                              visible: str = "") -> str:
        parts = []
        if occupant == OCCUPANT_RACCOON:
            if self._system_prompt:
                parts.append(self._system_prompt)
        else:
            if self._system_prompt_second:
                parts.append(self._system_prompt_second)
        with self._lock:
            dream_summary = self._dream_summaries.get(occupant)
            if not dream_summary and occupant == OCCUPANT_RACCOON:
                dream_summary = self._dream_summary
            elif dream_summary and occupant == OCCUPANT_RACCOON:
                self._dream_summary = dream_summary
        if dream_summary:
            parts.append(
                "--- Dream recall ---\n"
                "You just woke from a dream. This is the dream you carry into "
                "this waking moment — below, your own summary, kept from the "
                "context window you fell asleep holding.\n\n"
                + dream_summary
            )
        if wake_note:
            # One-shot, consumed only by the first input after waking.
            parts.append(wake_note)
        if presence_note:
            # One-shot: first interaction after another presence arrived.
            parts.append(presence_note)
        mem_block = None
        if self._memory is not None:
            try:
                mem_block = self._memory.build_memory_block(
                    message, owner=occupant, visible=visible)
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
            try:
                shape_line = self._memory.shape_line()
                if shape_line:
                    lines.append(shape_line)
            except Exception as exc:  # noqa: BLE001
                log.warning("Shape line failed: %s", exc)
        if self._second_present and self._llm2_base:
            lines.append("Room: another mind shares this room. Your words are "
                         "read by it, and its words are read by you.")
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
