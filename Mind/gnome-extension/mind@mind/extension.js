// extension.js — Mind GNOME Shell Extension (GNOME 45+, ESM)

import GLib     from 'gi://GLib';
import Gio      from 'gi://Gio';
import St       from 'gi://St';
import Clutter  from 'gi://Clutter';
import Pango    from 'gi://Pango';
import Meta     from 'gi://Meta';
import Shell    from 'gi://Shell';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main    from 'resource:///org/gnome/shell/ui/main.js';

const SENDMESSAGE_TIMEOUT_MS = 30000;

// ---------------------------------------------------------------------------
// D-Bus interface
// ---------------------------------------------------------------------------
const DBUS_IFACE = `<node>
  <interface name='com.mind.Daemon'>
    <method name='SendMessage'>
      <arg type='s' name='message'      direction='in'/>
      <arg type='s' name='token'        direction='in'/>
      <arg type='s' name='context_json' direction='in'/>
    </method>
    <method name='SetSystemPrompt'>
      <arg type='s' name='prompt' direction='in'/>
    </method>
    <method name='GetSystemPrompt'>
      <arg type='s' name='prompt' direction='out'/>
    </method>
    <method name='ClearHistory'/>
    <method name='GetHistory'>
      <arg type='s' name='history_json' direction='out'/>
    </method>
    <method name='SetNudgeConfig'>
      <arg type='b' name='enabled'      direction='in'/>
      <arg type='i' name='idle_minutes' direction='in'/>
    </method>
    <method name='TriggerNudge'/>
    <method name='Ping'>
      <arg type='s' name='pong' direction='out'/>
    </method>
    <signal name='StreamChunk'>
      <arg type='s' name='token'/>
      <arg type='s' name='text'/>
    </signal>
    <signal name='StreamThink'>
      <arg type='s' name='token'/>
      <arg type='s' name='text'/>
    </signal>
    <signal name='StreamDone'>
      <arg type='s' name='token'/>
    </signal>
    <signal name='NudgeStart'>
      <arg type='s' name='token'/>
    </signal>
        <signal name='StatusChanged'>
            <arg type='s' name='state'/>
            <arg type='s' name='detail'/>
        </signal>
    <signal name='StreamError'>
      <arg type='s' name='token'/>
      <arg type='s' name='error_msg'/>
    </signal>
  </interface>
</node>`;

// Interface info is not needed for proxy.call() + g-signal; pass null below.

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------
export default class MindExtension extends Extension {

    enable() {
        console.log('[Mind] extension.js build 2026-05-24T20:48 tooltipless');
        this._proxy           = null;
        this._proxySigs       = [];
        this._settingsChanged = null;
        this._window          = null;
        this._streaming       = false;
        this._currentToken    = null;
        this._streamLabel     = null;
        this._thinkLabel      = null;
        this._thinkToggle     = null;
        this._statusLabel     = null;
        this._statusState     = 'hanging-out';
        this._streamWatchdogId = 0;
        this._lastDaemonStartAttemptMs = 0;

        this._buildUI();
        this._connectDaemon();
        this._registerKeybinding();
    }

    disable() {
        // Keybinding
        Main.wm.removeKeybinding('mind-toggle');

        // UI
        if (this._window) {
            Main.layoutManager.removeChrome(this._window);
            this._window.destroy();
            this._window = null;
        }

        // D-Bus
        for (const id of this._proxySigs)
            this._proxy?.disconnect(id);
        this._proxySigs    = [];
        this._proxy        = null;
        this._streaming    = false;
        this._currentToken = null;
        this._streamLabel  = null;
        this._thinkLabel   = null;
        this._thinkToggle  = null;
        this._statusLabel  = null;
        this._statusState  = 'hanging-out';
        this._clearStreamWatchdog();


        // Settings
        if (this._settingsChanged) {
            this.getSettings().disconnect(this._settingsChanged);
            this._settingsChanged = null;
        }
    }

    // -----------------------------------------------------------------------
    // D-Bus
    // -----------------------------------------------------------------------
    _connectDaemon() {
        // Reconnecting: drop existing proxy signal handlers first.
        for (const id of this._proxySigs)
            this._proxy?.disconnect(id);
        this._proxySigs = [];
        this._proxy = null;

        Gio.DBusProxy.new(
            Gio.DBus.session,
            Gio.DBusProxyFlags.NONE,
            null,                    // no interface info needed for call()+g-signal
            'com.mind.Daemon',
            '/com/mind/Daemon',
            'com.mind.Daemon',
            null,                    // cancellable
            (source, result) => {
                try {
                    this._proxy = Gio.DBusProxy.new_finish(result);
                } catch (e) {
                    console.error(`[Mind] D-Bus proxy error: ${e}`);
                    return;
                }

                // Raw Gio.DBusProxy exposes ALL D-Bus signals through the
                // single 'g-signal' GObject signal — not individual methods.
                this._proxySigs.push(
                    this._proxy.connect('g-signal', (_proxy, _sender, signalName, params) => {
                        const unpacked = params.deepUnpack();
                        const token    = unpacked[0];

                        if (signalName === 'StreamChunk') {
                            const text = unpacked[1];
                            if (token !== this._currentToken) return;
                            this._bumpStreamWatchdog(token);
                            const cur = this._streamLabel?.get_text() ?? '';
                            this._streamLabel?.set_text(cur + text);
                            this._scrollToBottom();

                        } else if (signalName === 'StreamThink') {
                            const text = unpacked[1];
                            if (token !== this._currentToken) return;
                            this._bumpStreamWatchdog(token);
                            // Reveal the think section on the first token.
                            if (this._thinkLabel && !this._thinkLabel.visible) {
                                this._thinkLabel.show();
                                this._thinkToggle?.show();
                            }
                            const cur = this._thinkLabel?.get_text() ?? '';
                            this._thinkLabel?.set_text(cur + text);
                            this._scrollToBottom();

                        } else if (signalName === 'StreamDone') {
                            if (token !== this._currentToken) return;
                            this._clearStreamWatchdog();
                            // Format the completed answer with markdown → Pango
                            // while we still hold a reference to the label.
                            if (this._streamLabel) {
                                const finalText = this._streamLabel.get_text().trim();
                                if (!finalText) {
                                    // Command-only replies can resolve to empty text.
                                    // Remove the placeholder assistant bubble entirely.
                                    const col = this._streamLabel.get_parent();
                                    const row = col?.get_parent();
                                    row?.destroy();
                                } else {
                                    const markup = this._markdownToPango(finalText);
                                    try {
                                        this._streamLabel.clutter_text.set_markup(markup);
                                    } catch (_e) {
                                        // Invalid markup — plain text already displayed.
                                    }
                                }
                            }
                            this._streaming    = false;
                            this._currentToken = null;
                            this._streamLabel  = null;
                            this._thinkLabel   = null;
                            this._thinkToggle  = null;
                            this._setInputEnabled(true);

                        } else if (signalName === 'NudgeStart') {
                            // Daemon is about to stream an unprompted message.
                            // Show the bubble and create an empty assistant slot.
                            if (this._streaming) return; // already busy
                            this._streaming    = true;
                            this._currentToken = token;
                            this._showBubble();
                            try {
                                this._streamLabel = this._addMessage('assistant', '');
                            } catch (e) {
                                console.error(`[Mind] NudgeStart UI error: ${e}`);
                            }
                            this._setInputEnabled(false);
                            this._startStreamWatchdog(token);

                        } else if (signalName === 'StatusChanged') {
                            const state = unpacked[0];
                            const detail = unpacked[1] ?? '';
                            this._setStatus(state, detail);

                        } else if (signalName === 'StreamError') {
                            const errMsg = unpacked[1];
                            if (token !== this._currentToken) return;
                            this._clearStreamWatchdog();
                            this._streaming    = false;
                            this._currentToken = null;
                            if (this._streamLabel) {
                                this._streamLabel.add_style_class_name('mind-error');
                                this._streamLabel.set_text(`⚠ ${errMsg}`);
                                this._streamLabel = null;
                            }
                            this._thinkLabel   = null;
                            this._thinkToggle  = null;
                            this._setInputEnabled(true);
                            this._setStatus('hanging-out', 'idle');
                        }
                    }),
                );

                console.log('[Mind] D-Bus proxy connected.');

                if (!this._daemonHasOwner()) {
                    console.warn('[Mind] Proxy connected without daemon owner; attempting daemon start');
                    this._setStatus('hanging-out', 'daemon starting');
                    this._attemptStartDaemon();
                    GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 1, () => {
                        this._connectDaemon();
                        return GLib.SOURCE_REMOVE;
                    });
                    return;
                }

                // Avoid clobbering daemon-side prompt with empty settings on startup.
                this._pushSystemPrompt(false);
                this._pushNudgeConfig();
                this._setStatus('hanging-out', 'idle');
                this._settingsChanged = this.getSettings().connect(
                    'changed',
                    (settings, key) => {
                        if (key === 'system-prompt')    this._pushSystemPrompt(true);
                        if (key === 'nudge-enabled' || key === 'nudge-idle-minutes')
                            this._pushNudgeConfig();
                    },
                );
            },
        );
    }

    _pushNudgeConfig() {
        if (!this._proxy) return;
        const settings     = this.getSettings();
        const enabled      = settings.get_boolean('nudge-enabled');
        const idleMinutes  = settings.get_int('nudge-idle-minutes');
        this._proxy.call(
            'SetNudgeConfig',
            new GLib.Variant('(bi)', [enabled, idleMinutes]),
            Gio.DBusCallFlags.NONE,
            2000,
            null,
            null,
        );
    }

    _pushSystemPrompt(force = false) {
        if (!this._proxy) return;
        const prompt = this.getSettings().get_string('system-prompt');
        if (!force && !prompt.trim()) {
            console.log('[Mind] System prompt settings empty; preserving daemon prompt');
            return;
        }
        this._proxy.call(
            'SetSystemPrompt',
            new GLib.Variant('(s)', [prompt]),
            Gio.DBusCallFlags.NONE,
            2000,
            null,
            null,
        );
        console.log(`[Mind] System prompt pushed (${prompt.length} chars)`);
    }

    _daemonHasOwner() {
        try {
            const owner = this._proxy?.get_name_owner?.() ?? '';
            return Boolean(String(owner).trim());
        } catch (_e) {
            return false;
        }
    }

    _attemptStartDaemon() {
        const now = Date.now();
        if (now - this._lastDaemonStartAttemptMs < 5000)
            return;
        this._lastDaemonStartAttemptMs = now;
        try {
            const proc = Gio.Subprocess.new(
                ['systemctl', '--user', 'start', 'mind'],
                Gio.SubprocessFlags.NONE,
            );
            proc.wait_async(null, null);
        } catch (e) {
            console.warn(`[Mind] Could not start daemon via systemctl: ${e}`);
        }
    }

    _isDaemonUnavailableError(msg) {
        const s = String(msg ?? '').toLowerCase();
        return (
            s.includes('without an owner')
            || s.includes('serviceunknown')
            || s.includes('not activatable')
            || (s.includes('com.mind.daemon') && s.includes('error.serviceunknown'))
        );
    }

    _recoverFromDaemonUnavailable(token, msg) {
        console.warn(`[Mind] Daemon unavailable for ${token}; reconnecting (${msg})`);
        if (this._streamLabel) {
            this._streamLabel.add_style_class_name('mind-error');
            this._streamLabel.set_text('⚠ Daemon unavailable. Reconnecting…');
            this._streamLabel = null;
        }
        this._clearStreamWatchdog();
        this._streaming = false;
        this._currentToken = null;
        this._thinkLabel = null;
        this._thinkToggle = null;
        this._setInputEnabled(true);
        this._setStatus('hanging-out', 'daemon reconnecting');

        this._attemptStartDaemon();

        this._connectDaemon();
    }

    // -----------------------------------------------------------------------
    // UI construction
    // -----------------------------------------------------------------------
    _buildUI() {
        // ── Root container ────────────────────────────────────────────────
        this._window = new St.BoxLayout({
            style_class: 'mind-window',
            orientation: Clutter.Orientation.VERTICAL,
            width:       480,
            height:      580,
            visible:     false,
            reactive:    true,
            can_focus:   true,
            opacity:     0,
        });

        // ── Header ────────────────────────────────────────────────────────
        const header = new St.BoxLayout({
            style_class: 'mind-header',
            orientation: Clutter.Orientation.HORIZONTAL,
        });

        const title = new St.Label({
            style_class: 'mind-title',
            text:        '⬡  Mind',
            y_align:     Clutter.ActorAlign.CENTER,
            x_expand:    true,
        });

        const statusPill = new St.BoxLayout({
            style_class: 'mind-status',
            orientation: Clutter.Orientation.HORIZONTAL,
            x_align:     Clutter.ActorAlign.END,
            y_align:     Clutter.ActorAlign.CENTER,
        });

        const statusDot = new St.Label({
            style_class: 'mind-status-dot',
            text:        '•',
        });

        this._statusLabel = new St.Label({
            style_class: 'mind-status-label',
            text:        'Hanging out',
        });

        statusPill.add_child(statusDot);
        statusPill.add_child(this._statusLabel);

        const clearBtn = new St.Button({
            style_class: 'mind-icon-btn',
            label:       '↺',
            can_focus:   false,
        });
        clearBtn.connect('clicked', () => this._clearHistory());

        const closeBtn = new St.Button({
            style_class: 'mind-icon-btn',
            label:       '✕',
            can_focus:   false,
        });
        closeBtn.connect('clicked', () => this._hideBubble());

        header.add_child(title);
        header.add_child(statusPill);
        header.add_child(clearBtn);
        header.add_child(closeBtn);

        // ── Message scroll area ───────────────────────────────────────────
        this._scrollView = new St.ScrollView({
            style_class:          'mind-scroll',
            vscrollbar_policy:    St.PolicyType.AUTOMATIC,
            hscrollbar_policy:    St.PolicyType.NEVER,
            x_expand:             true,
            y_expand:             true,
            overlay_scrollbars:   true,
        });

        this._msgBox = new St.BoxLayout({
            style_class: 'mind-msgs',
            orientation: Clutter.Orientation.VERTICAL,
            x_expand:    true,
        });
        this._scrollView.set_child(this._msgBox);

        // ── Input bar ─────────────────────────────────────────────────────
        const inputRow = new St.BoxLayout({
            style_class: 'mind-input-row',
            orientation: Clutter.Orientation.HORIZONTAL,
        });

        this._entry = new St.Entry({
            style_class: 'mind-entry',
            hint_text:   'Message…',
            x_expand:    true,
            can_focus:   true,
        });
        this._entry.clutter_text.connect('activate', () => this._send());
        this._entry.clutter_text.connect('key-press-event', (_a, event) => {
            if (event.get_key_symbol() === Clutter.KEY_Escape) {
                this._hideBubble();
                return Clutter.EVENT_STOP;
            }
            return Clutter.EVENT_PROPAGATE;
        });

        this._sendBtn = new St.Button({
            style_class: 'mind-send-btn',
            label:       '↑',
            can_focus:   false,
        });
        this._sendBtn.connect('clicked', () => this._send());

        inputRow.add_child(this._entry);
        inputRow.add_child(this._sendBtn);

        // Assemble
        this._window.add_child(header);
        this._window.add_child(this._scrollView);
        this._window.add_child(inputRow);

        Main.layoutManager.addTopChrome(this._window);
        this._positionWindow();
    }

    // -----------------------------------------------------------------------
    // Keybinding — registered for Shell.ActionMode.ALL so it fires even when
    // the bubble's modal grab is active (SYSTEM_MODAL mode).
    // -----------------------------------------------------------------------
    _registerKeybinding() {
        Main.wm.addKeybinding(
            'mind-toggle',
            this.getSettings(),
            Meta.KeyBindingFlags.NONE,
            Shell.ActionMode.ALL,
            () => this._toggleBubble(),
        );
    }

    // -----------------------------------------------------------------------
    // Show / Hide
    // -----------------------------------------------------------------------
    _toggleBubble() {
        if (this._window.visible)
            this._hideBubble();
        else
            this._showBubble();
    }

    _showBubble() {
        this._positionWindow();
        this._window.show();
        this._window.ease({
            opacity:  255,
            duration: 160,
            mode:     Clutter.AnimationMode.EASE_OUT_QUAD,
        });

        // Route keyboard to the entry using the correct St API —
        // grab_key_focus() is scoped to the Clutter scene graph and does NOT
        // create a compositor-level Wayland keyboard grab.
        GLib.idle_add(GLib.PRIORITY_LOW, () => {
            this._entry.grab_key_focus();
            return GLib.SOURCE_REMOVE;
        });
    }

    _hideBubble() {
        this._window.ease({
            opacity:    0,
            duration:   160,
            mode:       Clutter.AnimationMode.EASE_OUT_QUAD,
            onComplete: () => this._window.hide(),
        });
    }

    _positionWindow() {
        const monitor = Main.layoutManager.primaryMonitor;
        if (!monitor) return;
        const x = monitor.x + monitor.width  - 496;
        const y = monitor.y + Math.floor((monitor.height - 600) / 2);
        this._window.set_position(x, y);
    }

    // -----------------------------------------------------------------------
    // Messaging
    // -----------------------------------------------------------------------
    _send() {
        if (this._streaming || !this._proxy) return;

        if (!this._daemonHasOwner()) {
            this._attemptStartDaemon();
            this._setStatus('hanging-out', 'daemon starting');
            this._addMessage('assistant', '⚠ Daemon is starting. Try again in a moment.');
            return;
        }

        const text = this._entry.get_text().trim();
        if (!text) return;

        this._entry.set_text('');
        this._setInputEnabled(false);
        this._streaming = true;

        // Generate UUID synchronously before the D-Bus call so _currentToken
        // is already set when the first StreamChunk signal arrives.
        const token = GLib.uuid_string_random();
        this._currentToken = token;
        this._startStreamWatchdog(token);

        // Add UI bubbles — kept separate from the proxy call so a UI error
        // can never prevent the message from reaching the daemon.
        try {
            this._addMessage('user', text);
            this._streamLabel = this._addMessage('assistant', '');
            this._scrollToBottom();
        } catch (e) {
            console.error(`[Mind] UI error in _send: ${e}`);
        }

        this._proxy.call(
            'SendMessage',
            new GLib.Variant('(sss)', [text, token, this._buildContext()]),
            Gio.DBusCallFlags.NONE,
            SENDMESSAGE_TIMEOUT_MS,
            null,
            (_proxy, result) => {
                try {
                    _proxy.call_finish(result);
                } catch (e) {
                    const msg = String(e?.message ?? e ?? '');
                    if (!this._daemonHasOwner() || this._isDaemonUnavailableError(msg)) {
                        this._recoverFromDaemonUnavailable(token, msg);
                        return;
                    }
                    // While this request token is still active, treat callback
                    // errors as non-fatal and keep waiting for stream signals.
                    // D-Bus ack timeouts and transient bus delays can occur even
                    // when the daemon later delivers chunks.
                    if (this._streaming && this._currentToken === token) {
                        console.warn(`[Mind] SendMessage ack issue for ${token}; awaiting stream signals (${msg})`);
                        this._setStatus('chatting', 'waiting for daemon');
                        if (this._streamLabel && !this._streamLabel.get_text().trim()) {
                            this._streamLabel.set_text('…waiting for daemon');
                        }
                        return;
                    }
                    console.error(`[Mind] SendMessage error: ${e}`);
                    if (this._streamLabel) {
                        this._streamLabel.add_style_class_name('mind-error');
                        this._streamLabel.set_text(`⚠ ${e.message}`);
                        this._streamLabel = null;
                    }
                    this._streaming    = false;
                    this._currentToken = null;
                    this._setInputEnabled(true);
                }
            },
        );
    }

    _buildContext() {
        // Snapshot current context synchronously — global.display.focus_window
        // returns the last Meta.Window with focus (a real app window), even
        // while the bubble's Clutter widget holds keyboard focus.
        const ctx = {};

        const now = GLib.DateTime.new_now_local();
        if (now)
            ctx.datetime = now.format('%A, %B %-d %Y %H:%M');

        try {
            const win = global.display.focus_window;
            if (win) {
                const title = win.get_title();
                if (title) ctx.window = title;
            }
        } catch (_e) {}

        return JSON.stringify(ctx);
    }

    _clearHistory() {
        this._proxy?.call('ClearHistory', null, Gio.DBusCallFlags.NONE, 2000, null, null);
        this._msgBox.get_children().forEach(c => c.destroy());
    }

    _setStatus(state, detail = '') {
        this._statusState = state || 'hanging-out';
        if (!this._statusLabel) return;

        const map = {
            'hanging-out': 'Hanging out',
            'chatting': 'Chatting',
            'napping': 'Napping',
            'sleeping': 'Sleeping',
            'idle': 'Hanging out',
        };

        const label = map[this._statusState] ?? 'Hanging out';
        this._statusLabel.set_text(label);
        this._statusLabel.remove_style_class_name('mind-status-hanging-out');
        this._statusLabel.remove_style_class_name('mind-status-chatting');
        this._statusLabel.remove_style_class_name('mind-status-napping');
        this._statusLabel.remove_style_class_name('mind-status-sleeping');
        this._statusLabel.add_style_class_name(`mind-status-${this._statusState}`);
    }

    _startStreamWatchdog(token) {
        this._clearStreamWatchdog();
        this._streamWatchdogId = GLib.timeout_add_seconds(
            GLib.PRIORITY_DEFAULT,
            180,
            () => {
                if (!this._streaming || token !== this._currentToken)
                    return GLib.SOURCE_REMOVE;

                console.warn(`[Mind] Stream watchdog timeout for token ${token}`);
                if (this._streamLabel) {
                    this._streamLabel.add_style_class_name('mind-error');
                    this._streamLabel.set_text('⚠ Response stalled. You can send another message.');
                }
                this._streaming = false;
                this._currentToken = null;
                this._streamLabel = null;
                this._thinkLabel = null;
                this._thinkToggle = null;
                this._setInputEnabled(true);
                this._setStatus('hanging-out', 'idle');
                this._streamWatchdogId = 0;
                return GLib.SOURCE_REMOVE;
            },
        );
    }

    _bumpStreamWatchdog(token) {
        if (!this._streaming || token !== this._currentToken)
            return;
        this._startStreamWatchdog(token);
    }

    _clearStreamWatchdog() {
        if (this._streamWatchdogId) {
            GLib.source_remove(this._streamWatchdogId);
            this._streamWatchdogId = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    _addMessage(role, text) {
        const isUser = role === 'user';

        const row = new St.BoxLayout({
            orientation: Clutter.Orientation.HORIZONTAL,
            x_expand: true,
            x_align:  isUser
                ? Clutter.ActorAlign.END
                : Clutter.ActorAlign.START,
        });

        if (isUser) {
            const bubble = new St.BoxLayout({
                style_class: 'mind-msg mind-msg-user',
                orientation: Clutter.Orientation.VERTICAL,
            });
            const label = new St.Label({
                text,
                x_expand: true,
            });
            label.clutter_text.line_wrap      = true;
            label.clutter_text.line_wrap_mode = Pango.WrapMode.WORD_CHAR;
            label.clutter_text.ellipsize      = Pango.EllipsizeMode.NONE;
            label.clutter_text.selectable     = true;
            bubble.add_child(label);
            row.add_child(bubble);
            this._msgBox.add_child(row);
            return label;
        }

        // ── Assistant bubble: vertical column with optional think section ──
        const col = new St.BoxLayout({
            style_class: 'mind-msg mind-msg-assistant',
            orientation: Clutter.Orientation.VERTICAL,
        });

        // Toggle button — hidden until the first reasoning token arrives.
        const thinkToggle = new St.Button({
            style_class: 'mind-think-toggle',
            label:       '▾ thinking',
            visible:     false,
            x_align:     Clutter.ActorAlign.START,
            reactive:    true,
            can_focus:   false,
        });

        // Reasoning text label — hidden until revealed.
        const thinkBody = new St.Label({
            style_class: 'mind-think',
            text:        '',
            visible:     false,
            x_expand:    true,
        });
        thinkBody.clutter_text.line_wrap      = true;
        thinkBody.clutter_text.line_wrap_mode = Pango.WrapMode.WORD_CHAR;
        thinkBody.clutter_text.ellipsize      = Pango.EllipsizeMode.NONE;
        thinkBody.clutter_text.selectable     = true;

        thinkToggle.connect('clicked', () => {
            thinkBody.visible   = !thinkBody.visible;
            thinkToggle.label   = thinkBody.visible ? '▾ thinking' : '▸ thinking';
        });

        // Answer label — always visible, fills the bubble.
        const answer = new St.Label({
            text:     text,
            x_expand: true,
        });
        answer.clutter_text.line_wrap      = true;
        answer.clutter_text.line_wrap_mode = Pango.WrapMode.WORD_CHAR;
        answer.clutter_text.ellipsize      = Pango.EllipsizeMode.NONE;
        answer.clutter_text.selectable     = true;

        col.add_child(thinkToggle);
        col.add_child(thinkBody);
        col.add_child(answer);
        row.add_child(col);
        this._msgBox.add_child(row);

        // Expose to signal handler via instance fields.
        // These are overwritten each time a new assistant message starts.
        this._thinkToggle = thinkToggle;
        this._thinkLabel  = thinkBody;

        return answer;
    }

    _scrollToBottom() {
        // Queue a scroll on the next frame using GLib.idle_add so this
        // method is always synchronous and can never throw into callers.
        GLib.idle_add(GLib.PRIORITY_LOW, () => {
            try {
                const adjustment = this._scrollView?.vscroll?.adjustment;
                if (adjustment)
                    adjustment.value = Math.max(0, adjustment.upper - adjustment.page_size);
            } catch (_e) { /* ignore any version-specific API mismatch */ }
            return GLib.SOURCE_REMOVE;
        });
    }

    // Converts a subset of Markdown to Pango markup.
    // Code spans/blocks are extracted first so bold/italic patterns cannot
    // fire inside them; they are restored after all other transforms.
    _markdownToPango(raw) {
        const esc = t =>
            t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // ── Step 1: protect code regions ────────────────────────────────
        const saved = [];
        let   n     = 0;
        let   s     = raw;

        // Fenced code blocks  ```lang\n...\n```
        s = s.replace(/```(?:\w*)\n?([\s\S]*?)```/g, (_, code) => {
            const mark = `\x01B${n++}\x01`;
            saved.push(
                `<span foreground="#8fbcbb"><tt>${esc(code.trim())}</tt></span>`
            );
            return mark;
        });

        // Inline code  `code`
        s = s.replace(/`([^`\n]+)`/g, (_, code) => {
            const mark = `\x01B${n++}\x01`;
            saved.push(`<tt>${esc(code)}</tt>`);
            return mark;
        });

        // ── Step 2: escape remaining text for Pango ──────────────────────
        s = esc(s);

        // ── Step 3: markdown transforms ──────────────────────────────────
        // Bold + italic  ***text***
        s = s.replace(/\*\*\*([^*]+?)\*\*\*/g, '<b><i>$1</i></b>');
        // Bold  **text**
        s = s.replace(/\*\*([^*]+?)\*\*/g, '<b>$1</b>');
        // Italic  *text*  (remaining single-star after bold consumed **)
        s = s.replace(/\*([^*\n]+)\*/g, '<i>$1</i>');
        // Italic  _text_  (only when surrounded by non-underscore)
        s = s.replace(/(?<![\w])_([^_\n]+)_(?![\w])/g, '<i>$1</i>');

        // Headings  (must come before bullet so # is not swallowed)
        s = s.replace(/^### (.+)$/gm, '<b>$1</b>');
        s = s.replace(/^## (.+)$/gm,
            '<span size="large"><b>$1</b></span>');
        s = s.replace(/^# (.+)$/gm,
            '<span size="x-large"><b>$1</b></span>');

        // Bullets  - item  or  * item  (leading whitespace ok)
        s = s.replace(/^[ \t]*[-*] (.+)$/gm, '  \u2022 $1');

        // Numbered lists  — keep number, just indent slightly
        s = s.replace(/^[ \t]*(\d+)\. (.+)$/gm, '  $1. $2');

        // Horizontal rule  ---
        s = s.replace(/^---+$/gm, '\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500');

        // ── Step 4: restore protected code regions ───────────────────────
        for (let i = 0; i < saved.length; i++)
            s = s.replace(`\x01B${i}\x01`, saved[i]);

        return s;
    }

    _setInputEnabled(enabled) {
        this._entry.reactive    = enabled;
        this._entry.can_focus   = enabled;
        this._sendBtn.reactive  = enabled;
        if (enabled && this._window?.visible) {
            GLib.idle_add(GLib.PRIORITY_LOW, () => {
                this._entry.grab_key_focus();
                return GLib.SOURCE_REMOVE;
            });
        }
    }
}
