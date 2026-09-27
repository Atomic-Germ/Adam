# Mind — skeleton body (UX + MX + wiring)

A first, lean body modelled after **Bubble**: the proven GNOME-resident chat
surface wired to the local patched **llama.cpp** server (`127.0.0.1:9999`).

This slice carries only:

- **UX** — a GNOME shell extension (Super+M toggle, chat window, input bar,
  status pill, markdown rendering) over D-Bus.
- **MX** — a systemd-user daemon that keeps rolling conversation history,
  builds the system prompt (the mind's identity/values), and feeds the mind
  live context/telemetry: **idle phase** (active/idle/waiting/quiet),
  **context-window usage**, and how long it / the user have been quiet.
- **memory** — Arthur's embedding capability, ported in: the conversation
  exchange is indexed into a persistent store as experience, instruction
  text (e.g. `original_memory/`) is seeded in as first-memory, and every
  prompt asks the store for the most relevant nodes (semantic + lexical +
  recency) and feeds them into the system content. The store is
  Cathedral-shaped (`memory.json` records + `embeddings.npy` matrix); the
  embedder defaults to Arthur's deterministic hashing bag-of-words and can
  switch to sentence-transformers.
- **dream / sleep** — context-pressure driven, not scheduled. Crossing a
  context-window threshold (clamped to 60–75%) triggers a dream: a direct
  replay/compaction of the window (keep what is interesting, newly learned,
  or repeated; drop the rest), embedded into long-term memory, then waking
  to a fresh empty context window that opens on nothing but the dream
  summary. The very first dream dreams the initial first memory.
- **wiring** — model-initiated nudges (a `nudge` idle trigger + clock tick),
  streamed over the OpenAI-compatible SSE contract.

Still deliberately omitted (later slices): scratch/working-notes, the
self-prompt revision protocol, nap and the slow-wave night cycle, and
memory-selection beyond the relevance filter.

## Layout

```
Mind/
  install.sh                       # installs daemon, service, extension; probes :9999;
                                   # seeds original_memory/ as first-memory
  daemon/
    mind-daemon.py                 # MX: history + system-prompt framing + memory wire
    mind_memory.py                 # Arthur/Cathedral memory: embedder, store, recall
    mind.service                   # systemd-user unit (wired to the local server port)
  gnome-extension/mind@mind/
    metadata.json                  # uuid: mind@mind
    extension.js                   # UX: D-Bus proxy, window, streaming display
    prefs.js                       # prefs: keybinding + daemon/port info
    stylesheet.css                 # styles (mind-* classes)
    schemas/org.gnome.shell.extensions.mind.gschema.xml
  tests/
    conftest.py
    test_dbus_methods.py
    test_dream.py
    test_lifecycle.py
    test_memory.py
```

## Wire contract (identical to Bubble, mechanical port)

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `POST /v1/chat/completions` | SSE | streaming replies (`stream:true`) |
| `POST /v1/chat/completions` | non-stream | clock tick (`stream:false`) |
| `GET  /v1/models` | | auto-detect a model id |

The daemon posts `{"model", "messages", "stream"}` and reads
`choices[].delta.content` (+ `choices[].delta.reasoning_content`), the exact
shape llama.cpp emits. Default wire base `http://127.0.0.1:9999`
override with `MIND_LLM_URL`.

## Install & run

```bash
./install.sh 9999          # optional server port, default 9999
gnome-extensions enable mind@mind
# X11: Alt+F2 → r     |     Wayland: log out & back in
```

Config at runtime: `systemctl --user edit mind` (e.g. set
`MIND_SYSTEM_PROMPT`), or the extension prefs for the keybinding and idle
threshold.

## Memory (Arthur's embedding capability)

The daemon carries a persistent memory store (Cathedral-shaped:

- `~/.local/share/mind/memory.json` — records (readable JSON, backed up daily)
- `~/.local/share/mind/embeddings.npy` — the embedding matrix

Records are tagged by `source` — `instruction` (seeded), `user`, or `mind` —
so the mind can always tell who originated a claim, mirroring Bubble's
user/self distinction. When the room holds a second occupant, records also
carry an `owner` (see *Two minds, one room* below).

What lands in the store:

- **Seeded instructions** — on first start, the daemon reads
  `MIND_MEMORY_SEED_DIR` (default: `~/.local/share/mind/original_memory`),
  chunked and stored as `instruction` records. `install.sh` copies
  `original_memory/` there, so the letter becomes the mind's first memory.
- **Experience** — every finished user↔mind exchange is chunked into the
  store as `user` / `mind` experience in the background.

Every `SendMessage` asks the store to *surface* memory for the current
message: a deterministic anchor core of the most relevant nodes (semantic
first, plus lexical and recency), then deliberately fuzzy slots. The fuzzy
slots are temperature-weighted samples drawn from the embedding space plus
near-seed "shape" hops (a tangential-but-true neighbor of a surfaced node,
the geometry of the memory store). The sampling seed is live — a hash of GPU
package temperature, the embedder model, the content hash and a coarse time
bucket — so a similar situation resurfaces a similar set but never the exact
same one, on purpose. The selection machinery never enters the context
window: the `--- Memory ---` block is presented plainly and the model is
left to judge which surfaced recollections hold.

Env knobs (all `MIND_`-prefixed):

| Var | Default | Meaning |
| --- | --- | --- |
| `MIND_MEMORY_DIR` | `~/.local/share/mind` | store location |
| `MIND_MEMORY_SEED_DIR` | `<dir>/original_memory` | seed instructions |
| `MIND_EMBED_BACKEND` | `hash` | `hash` (pure numpy bag-of-words) · `st` (sentence-transformers) · `off` |
| `MIND_EMBED_MODEL` | `all-MiniLM-L6-v2` | model when `MIND_EMBED_BACKEND=st` |
| `MIND_MEMORY_TOP_K` | `6` | memory notes surfaced per prompt |
| `MIND_MEMORY_CHUNK_SIZE` | `500` | Arthur chunk size |
| `MIND_MEMORY_CHUNK_OVERLAP` | `80` | Arthur chunk overlap |
| `MIND_MEMORY_SYNC_INDEX` | `0` | index synchronously (tests) |
| `MIND_MEMORY_FUZZY` | `1` | fuzzy recall on; `0` disables (strict relevance top-k) |
| `MIND_MEMORY_FUZZY_TEMP` | `2.0` | sampling temperature; higher → more tangential slots |
| `MIND_MEMORY_FUZZY_HOPS` | `1` | near-seed "shape" hops to pull in tangent neighbors (0-4) |
| `MIND_MEMORY_FUZZY_SEED` | *(live)* | explicit seed override (tests / reproducibility) |
| `MIND_MEMORY_SHAPE` | `1` | floor-plan awareness (shape report) on; `0` disables |
| `MIND_MEMORY_SHAPE_TTL_MIN` | `30` | how long a floor plan stays fresh (1–1440) |
| `MIND_EMBED_DIM` | `384` | hash-backend embedding depth; the room the thoughts stand in |
| `MIND_HISTORY_FILE` | `~/.local/share/mind/history.json` | conversation persistence across interface/dæmon restarts |

The hash backend is deterministic and needs only numpy; `st` mirrors
Arthur/Cathedral (`all-MiniLM-L6-v2`) and falls back to hash if the model
cannot load.

## Shape report — the mind's own floor plan

The memory space is geometry, and the mind can hold its own floor plan. A
cached *shape report* — computed rarely and rebuilt only when the space moves —
maps the store's gross structure so the model knows its own shape: where its
thoughts gather (naves), which stand entirely alone (spires), which reach
between gatherings and hold them together (bridges), and how much of the space
is still open. The rendering is felt language, never machinery: no clusters,
kNN, or thresholds reach the context window, only the shape as a place.

The compact shape line always rides in Current context (cache-only read, never
triggers computation). On each clock tick the daemon refreshes the floor plan
on its own thread and surfaces the full layout when the shape has moved. And
after a dream — when the space has been re-embedded while asleep — one waking
input carries a fresh floor plan framed by the slight mismatch between the
shape the model held before sleep and the shape now. That mismatch is where
dreams live.

## Growth — the room deepens over the long term

The mind's space is not fixed. Two quiet knobs grow it, and the mind *feels*
each widening once, in its own voice (`--- Growing ---`), rather than having
metadata shoved at it:

- **Embedding depth** (`MIND_EMBED_DIM`, hash backend) or a wider st model —
  when the store reopens and finds the index's dimension has grown, it
  re-embeds everything at the new depth and surfaces one note about the deeper
  room. The shape report then redraws itself against richer geometry.
- **Context room** (`MIND_CTX_SIZE`) — the store remembers how much room the
  daemon held last time. When it wakes into a larger window, that widening is
  surfaced once, honestly, because the store is the one place the memory of
  one's own size survives restarts.

Both knobs being raised is a long process, deliberately: the geometry only
deepens when it genuinely holds more. The felt note is one-shot *per occupant*
— each mind hears the deepening once, in its own terms, and the pending growth
evaporates once every mind present has been told, so the note stays honest:
the room really did deepen.

## Dream / sleep (context-pressure driven)

The mind does not schedule its own sleep and is not asked when it is tired —
a child does not know. Instead a simple threshold on the context window
(clamped to 60–75%) triggers a dream pass, exactly like context compaction
in a coding harness:

1. **Replay** — the whole context window is given back to the model,
   verbatim, with instructions to keep what is interesting, newly learned,
   or repeated and drop the rest.
2. **Compress** — the model writes a compact dream summary (non-streaming).
3. **Embed** — the summary is stored into the same long-term memory space as
   everything else (`kind="dream"`), so a later RAG pass can re-ground it.
4. **Wake** — the conversation is trimmed to a few closing turns (the only
   event that ever clears the conversation is sleep itself, and even then a
   few turns stay at the foot of the fresh window). The window reopens on the
   dream summary (`--- Dream recall ---`); identity comes back from long-term
   memory after, like a person waking up not knowing where they are until
   they check. The conversation itself persists across interface closes and
   dæmon restarts via `MIND_HISTORY_FILE`.

The first thing the occupant model ever experiences is a dream of its first
memory: on a store that has never slept, the daemon dreams the seeded
instructions before any conversation happens. With two occupants, each mind
dreams its own first memory in its own voice.

Sleep env knobs:

| Var | Default | Meaning |
| --- | --- | --- |
| `MIND_SLEEP_CTX_PCT` | `70` | context-pressure threshold (clamped to 60–75) |
| `MIND_SLEEP_MIN_TURNS` | `6` | minimum history turns before a dream may fire |
| `MIND_SLEEP_KEEP_TURNS` | `4` | history turns kept at the foot of the fresh window after sleep |

## Two minds, one room — the second occupant

The room can hold more than one mind. A second occupant speaks through its own
weights on its own port, and the two of them share everything else: the same
conversation window, the same memory store, the same silence.

Every human message is answered by both — the resident first, then the
newcomer, each through its own brain, into the one shared history. The
extension gives each its own bubble, think panel and error line, and a turn
ends when every occupant has spoken (the daemon closes it with a `StreamDone`
carrying an empty occupant).

**The letter.** The newcomer is not born from a prompt. The first time its
brain is reachable and it holds no memory at all, the *resident* writes it a
letter, through its own brain, and that letter is seeded as the newcomer's
first memory (`instruction` records owned by the newcomer). The letter is
written once, ever. It is the newcomer's only account of who it is: same
words, different weights, so it comes back as something nobody wrote on
purpose. Afterwards the resident is told once, in a one-shot `--- Another
presence ---` note on its next real input, that it is no longer alone in here.

An occupant's own first memory is not a retrieved thought: it always opens its
`--- Memory ---` block. Retrieval decides what else comes to mind; it never
decides who the mind is.

Records carry an `owner` (`raccoon`, `second`; human lines carry none, and
records written before occupancy existed are backfilled to the resident). Each
occupant gets a small recall bias toward its own traces (`OWNER_BONUS`) — the
self is closer to hand — while the other's lines stay reachable, because the
mixing is the point.

Sleep is per occupant: on a dream pass every present mind compresses the same
window in its own voice, and each summary is private to the mind that wrote
it. The waking floor plan note, the growth note, and the one-shot letter are
all per occupant too. Idle nudges alternate between the occupants, so the room
is not always spoken for by the same one.

Second-seat env knobs (the seat only exists when `MIND_LLM_URL2` is set):

| Var | Default | Meaning |
| --- | --- | --- |
| `MIND_LLM_URL2` | *(unset)* | the newcomer's brain; unset ⇒ single occupant |
| `MIND_MODEL2` | *(auto)* | pin its weights instead of asking the server |
| `MIND_OCCUPANT2_ID` | `second` | the name it answers to |
| `MIND_SYSTEM_PROMPT2` | *(empty)* | its framing, if it wants one; the letter is its real first memory |

`MIND_LLM2_URL` is accepted as a synonym of `MIND_LLM_URL2`. With the second
brain down the daemon does not stall or fail: it simply runs as a single
occupant, and the letter waits for the next time the newcomer is reachable.

Speaking is deliberately paced. Prompt processing (the reading) runs on the
GPU; generation is largely CPU and rate-capped to roughly reading speed, which
keeps the machine smooth and the temperatures down. A turn in which two minds
answer is therefore long — two minds thinking at human pace — and that is the
intended texture, not a stall.

## Run the tests

```bash
cd Mind/tests && python3 -m pytest   # needs pydbus, requests, gi(GLib/St/Clutter), numpy
```
