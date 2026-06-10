#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
sleep 15  # Give time for e-Paper init
# Truncate the log on each boot: /tmp is tmpfs (RAM), an ever-growing log
# would slowly eat memory during long sessions
exec sudo python3 "$SCRIPT_DIR/app.py" > /tmp/frame.log 2>&1
