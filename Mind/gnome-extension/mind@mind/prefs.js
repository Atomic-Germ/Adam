// prefs.js — Mind Extension Preferences (GNOME 45+)

import Adw from 'gi://Adw';
import Gtk from 'gi://Gtk';
import Gio from 'gi://Gio';

import {
    ExtensionPreferences,
    gettext as _,
} from 'resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js';

export default class MindPreferences extends ExtensionPreferences {

    fillPreferencesWindow(window) {
        const settings = this.getSettings();

        const page = new Adw.PreferencesPage({
            title:     _('General'),
            icon_name: 'preferences-system-symbolic',
        });

        // ── Keyboard shortcut ──────────────────────────────────────────────
        const kbGroup = new Adw.PreferencesGroup({
            title:       _('Keyboard Shortcut'),
            description: _('Shortcut to toggle the chat window.'),
        });

        const currentBinding = settings.get_strv('mind-toggle')[0] ?? '(none)';

        const kbRow = new Adw.ActionRow({
            title:    _('Toggle Mind'),
            subtitle: currentBinding,
        });

        const copyBtn = new Gtk.Button({
            label:        _('Copy gsettings command'),
            valign:       Gtk.Align.CENTER,
            css_classes:  ['flat'],
        });
        copyBtn.connect('clicked', () => {
            const cmd =
                `gsettings set org.gnome.shell.extensions.mind ` +
                `mind-toggle "['<Super>b']"`;
            const display = copyBtn.get_display();
            display.get_clipboard().set(cmd);
        });
        kbRow.add_suffix(copyBtn);

        kbGroup.add(kbRow);

        const hint = new Adw.ActionRow({
            title: _('To change the shortcut, run in a terminal:'),
        });
        const hintLabel = new Gtk.Label({
            label:        `gsettings set org.gnome.shell.extensions.mind mind-toggle "['&lt;Super&gt;b']"`,
            wrap:         true,
            use_markup:   true,
            halign:       Gtk.Align.START,
            css_classes:  ['caption', 'dim-label'],
            selectable:   true,
        });
        hint.add_suffix(hintLabel);
        kbGroup.add(hint);

        // ── Daemon info ────────────────────────────────────────────────────
        const daemonGroup = new Adw.PreferencesGroup({
            title:       _('Daemon'),
            description: _('The Mind daemon bridges the extension to llama.cpp.'),
        });

        const daemonRow = new Adw.ActionRow({
            title:    _('Service name'),
            subtitle: _('mind.service  (systemd user service)'),
        });

        const startBtn = new Gtk.Button({
            label:       _('systemctl --user start mind'),
            valign:      Gtk.Align.CENTER,
            css_classes: ['flat'],
        });
        startBtn.connect('clicked', () => {
            try {
                const proc = Gio.Subprocess.new(
                    ['systemctl', '--user', 'start', 'mind'],
                    Gio.SubprocessFlags.NONE,
                );
                proc.wait_async(null, null);
            } catch (e) {
                console.error(`[Mind prefs] ${e}`);
            }
        });
        daemonRow.add_suffix(startBtn);
        daemonGroup.add(daemonRow);

        const portRow = new Adw.ActionRow({
            title:    _('llama.cpp port'),
            subtitle: _('Default 52625 — override with $FLM_SERVE_PORT in the service environment.'),
        });
        daemonGroup.add(portRow);

        page.add(kbGroup);
        page.add(daemonGroup);
        window.add(page);
    }
}
