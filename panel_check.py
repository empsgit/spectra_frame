#!/usr/bin/env python3
"""Standalone e-paper panel diagnostic for the Waveshare 13.3" Spectra 6 HAT.

Probes the busy pin directly, bypassing the frame app. Run with the
service stopped:

    sudo systemctl stop spectra_frame.service
    sudo python3 panel_check.py
    sudo systemctl start spectra_frame.service

Interpretation:
  - "OK: panel responded"      -> wiring is fine, panel controller alive
  - busy never changes at all  -> controller gets no power/reset/SPI:
                                  re-seat the FFC ribbon cable to the panel
                                  (contacts fully in, latch closed, right
                                  orientation) and the HAT on the 40-pin
                                  header
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib'))
import epdconfig

BUSY = epdconfig.EPD_BUSY_PIN
RST = epdconfig.EPD_RST_PIN

print("module_init (SPI/GPIO setup)...")
epdconfig.module_init()

print("busy pin before reset: %d  (0=busy/stuck, 1=idle)" % epdconfig.digital_read(BUSY))

print("hardware reset...")
for v in (1, 0, 1, 0, 1):
    epdconfig.digital_write(RST, v)
    time.sleep(0.05)

t0 = time.time()
last = None
transitions = 0
ok = False
while time.time() - t0 < 30:
    v = epdconfig.digital_read(BUSY)
    if v != last:
        print("t=%6.2fs  busy=%d" % (time.time() - t0, v))
        if last is not None:
            transitions += 1
        last = v
    if v == 1 and time.time() - t0 > 0.5:
        ok = True
        break
    time.sleep(0.01)

if ok:
    print("OK: panel responded, busy released %.2fs after reset" % (time.time() - t0))
elif transitions == 0:
    print("FAIL: busy line NEVER changed within 30s.")
    print("The controller is not running: check the FFC ribbon cable between")
    print("the panel and the HAT (orientation, fully inserted, latch closed)")
    print("and that the HAT sits firmly on the 40-pin header.")
else:
    print("FAIL: busy line toggled but never released within 30s.")
    print("Controller alive but stuck mid-init: likely power supply sag or a")
    print("partially seated FFC cable.")

epdconfig.module_exit()
print("done")
