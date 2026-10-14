#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo="$(pwd)"
mkdir -p "$HOME/.config/systemd/user" "$HOME/Desktop"
cat > "$HOME/.config/systemd/user/driving-school.service" <<EOF
[Unit]
Description=Dave's Garage GPU Driving School
After=network-online.target
[Service]
Type=simple
WorkingDirectory=$repo
ExecStart=$repo/scripts/run.sh --max-load
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
[Install]
WantedBy=default.target
EOF
cat > "$HOME/Desktop/Driving School.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Dave's Driving School
Comment=Watch a quarter-million cars learn to drive on the GB300
Exec=$repo/scripts/watch.sh
Icon=applications-games
Terminal=false
Categories=Game;Education;
EOF
if mkdir -p "$HOME/.local/share/applications" 2>/dev/null; then
    cp "$HOME/Desktop/Driving School.desktop" "$HOME/.local/share/applications/driving-school.desktop"
fi
chmod +x "$HOME/Desktop/Driving School.desktop"
gio set "$HOME/Desktop/Driving School.desktop" metadata::trusted true 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now driving-school.service
printf 'Driving school is available at http://127.0.0.1:8780/\n'
