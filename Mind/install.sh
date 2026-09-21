#!/usr/bin/env bash
# install.sh — Mind installation script (skeleton: UX + MX + llama.cpp wiring)
# Usage: ./install.sh [port]     # port defaults to 9999 (local patched llama.cpp server)
#
# Installs:
#   * the mind daemon  -> ~/.local/lib/mind/mind-daemon.py
#   * the systemd user service -> ~/.config/systemd/user/mind.service
#   * the GNOME shell extension -> ~/.local/share/gnome-shell/extensions/mind@mind
#
# The daemon talks to the local patched llama.cpp OpenAI-compatible server,
# default http://127.0.0.1:9999, over POST /v1/chat/completions (SSE).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-9999}"

# Colours
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}▶${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
die()   { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DAEMON_LIB="$HOME/.local/lib/mind"
DAEMON_SRC="$SCRIPT_DIR/daemon/mind-daemon.py"
SERVICE_SRC="$SCRIPT_DIR/daemon/mind.service"
SERVICE_DST="$HOME/.config/systemd/user/mind.service"
EXT_SRC="$SCRIPT_DIR/gnome-extension/mind@mind"
EXT_DST="$HOME/.local/share/gnome-shell/extensions/mind@mind"

# ---------------------------------------------------------------------------
# GNOME Shell version
# ---------------------------------------------------------------------------
info "Checking GNOME Shell version…"
SHELL_VER=$(gnome-shell --version 2>/dev/null | grep -oP '\d+' | head -1 || echo "0")
if [[ "$SHELL_VER" -lt 45 ]]; then
    die "GNOME Shell 45+ required (found version $SHELL_VER). Aborting."
fi
info "GNOME Shell $SHELL_VER — OK"

# ---------------------------------------------------------------------------
# Python daemon dependencies (pydbus, requests)
# ---------------------------------------------------------------------------
info "Installing Python daemon dependencies…"
if ! python3 -c "import pydbus" 2>/dev/null; then
    python3 -m pip install --user --quiet pydbus requests
    info "pip packages installed."
else
    info "pydbus already available."
fi

# ---------------------------------------------------------------------------
# Confirm the local llama.cpp server is reachable on the chosen port
# ---------------------------------------------------------------------------
info "Checking for local llama.cpp server on port ${PORT}…"
if command -v curl >/dev/null 2>&1; then
    if ! curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/health" >/dev/null 2>&1; then
        warn "No response from http://127.0.0.1:${PORT}/v1/health"
        warn "The daemon will retry, but chat replies will fail until it is up."
    else
        info "llama.cpp server reachable on port ${PORT}."
    fi
else
    warn "curl not found; skipping health probe (install later if needed)."
fi

# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------
info "Installing daemon to ${DAEMON_LIB}…"
mkdir -p "${DAEMON_LIB}"
cp "${DAEMON_SRC}" "${DAEMON_LIB}/mind-daemon.py"
chmod +x "${DAEMON_LIB}/mind-daemon.py"

# ---------------------------------------------------------------------------
# systemd user service  — wire the wire port into the unit
# ---------------------------------------------------------------------------
info "Installing systemd user service…"
mkdir -p "$(dirname "${SERVICE_DST}")"
sed "s#MIND_LLM_URL=#MIND_LLM_URL=#g" "${SERVICE_SRC}" > "${SERVICE_DST}"
# Inject the current server port (fallback to 9999).
sed -i "s/# PassEnvironment=MIND_LLM_URL.*/Environment='MIND_LLM_URL=http:\/\/127.0.0.1:${PORT}'\n# PassEnvironment=MIND_MODEL MIND_SYSTEM_PROMPT/" "${SERVICE_DST}"
systemctl --user daemon-reload
systemctl --user enable --now mind.service
info "mind.service enabled and started."

# ---------------------------------------------------------------------------
# GNOME extension
# ---------------------------------------------------------------------------
info "Installing GNOME extension…"
mkdir -p "${EXT_DST}"
cp -r "${EXT_SRC}/." "${EXT_DST}/"

# Compile GSettings schema
info "Compiling GSettings schemas…"
mkdir -p "${EXT_DST}/schemas"
glib-compile-schemas "${EXT_DST}/schemas/"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}✔ Installation complete!${NC}"
echo ""
echo "  Next steps:"
echo "    1. Restart GNOME Shell (X11: Alt+F2 → 'r'  |  Wayland: log out & back in)"
echo "    2. Enable the extension:"
echo "       gnome-extensions enable mind@mind"
echo ""
echo "  Keyboard shortcut: Super+M  (change via gsettings)"
echo ""
echo "  Service status:    systemctl --user status mind"
echo "  Service logs:      journalctl --user -u mind -f"
echo "  Wire port:         Environment='MIND_LLM_URL=http://127.0.0.1:9999'"
echo ""
