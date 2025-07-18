#!/bin/bash
sleep 15  # Give time for e-Paper init
exec sudo python3 /home/zero/spectra_frame/app.py >> /tmp/frame.log 2>&1