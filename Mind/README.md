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
- **wiring** — model-initiated nudges (a `nudge` idle trigger + clock tick),
  streamed over the OpenAI-compatible SSE contract.

Still deliberately omitted (later slices): scratch/working-notes, the
self-prompt revision protocol, summarization, dream/nap/slow-wave night
cycle, and memory-selection beyond the relevance filter.

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
user/self distinction.

What lands in the store:

- **Seeded instructions** — on first start, the daemon reads
  `MIND_MEMORY_SEED_DIR` (default: `~/.local/share/mind/original_memory`),
  chunked and stored as `instruction` records. `install.sh` copies
  `original_memory/` there, so the letter becomes the mind's first memory.
- **Experience** — every finished user↔mind exchange is chunked into the
  store as `user` / `mind` experience in the background.

Every `SendMessage` asks the store for the most relevant nodes to the current
message (semantic similarity first, plus lexical and recency signals) and
injects them into the system content as a `--- Memory ---` block. Embeddings
are the key: without them retrieval degrades to lexical + recency only.

Env knobs (all `MIND_`-prefixed):

| Var | Default | Meaning |
| --- | --- | --- |
| `MIND_MEMORY_DIR` | `~/.local/share/mind` | store location |
| `MIND_MEMORY_SEED_DIR` | `<dir>/original_memory` | seed instructions |
| `MIND_EMBED_BACKEND` | `hash` | `hash` (pure numpy bag-of-words) · `st` (sentence-transformers) · `off` |
| `MIND_EMBED_MODEL` | `all-MiniLM-L6-v2` | model when `MIND_EMBED_BACKEND=st` |
| `MIND_MEMORY_TOP_K` | `6` | memory nodes surfaced per prompt |
| `MIND_MEMORY_CHUNK_SIZE` | `500` | Arthur chunk size |
| `MIND_MEMORY_CHUNK_OVERLAP` | `80` | Arthur chunk overlap |
| `MIND_MEMORY_SYNC_INDEX` | `0` | index synchronously (tests) |

The hash backend is deterministic and needs only numpy; `st` mirrors
Arthur/Cathedral (`all-MiniLM-L6-v2`) and falls back to hash if the model
cannot load.

## Run the tests

```bash
cd Mind/tests && python3 -m pytest   # needs pydbus, requests, gi(GLib/St/Clutter), numpy
```
