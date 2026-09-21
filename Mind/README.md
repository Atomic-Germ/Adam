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
- **wiring** — model-initiated nudges (a `nudge` idle trigger + clock tick),
  streamed over the OpenAI-compatible SSE contract.

Deliberately omitted (Arthur pieces, come later): memory/RAG/embeddings,
scratch/working-notes, the self-prompt revision protocol, summarization,
dream/nap/slow-wave night cycle, and memory-selection.

## Layout

```
Mind/
  install.sh                       # installs daemon, service, extension; probes :9999
  daemon/
    mind-daemon.py                 # MX: history + system-prompt framing + nudge/telemetry
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

## Run the tests

```bash
cd Mind/tests && python3 -m pytest   # needs pydbus, requests, gi(GLib/St/Clutter)
```
