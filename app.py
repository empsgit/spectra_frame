#!/usr/bin/env python3

import sys
import os
import time
import threading
import io
import base64
import json
import random
import queue
import copy
import numpy as np
import RPi.GPIO as GPIO

from flask import Flask, request, render_template_string, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance

# Add local lib directory for Waveshare e-paper driver
libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)
import epd13in3E  # Waveshare 13.3" E-Paper module

# ==== Directories & Config ====
BASE_DIR    = os.path.dirname(os.path.realpath(__file__))
picdir      = os.path.join(BASE_DIR, 'pic')
single_dir  = os.path.join(picdir, 'single')
pool_dir    = os.path.join(picdir, 'pool')
config_path = os.path.join(BASE_DIR, 'config.json')

GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.OUT, initial=GPIO.LOW)


thumbs_dir = os.path.join(picdir, 'thumbs')

os.makedirs(single_dir, exist_ok=True)
os.makedirs(pool_dir, exist_ok=True)
os.makedirs(thumbs_dir, exist_ok=True)

default_config = {
    "mode":          "single",
    "single_image":  "",
    "pool_images":   [],
    "dithering":     "floyd-steinberg",
    "update_count":  0,
    "fit_mode":      "pad"  # pad | zoom | stretch
}

# Config is cached in memory to avoid re-reading the SD card on every
# request; the lock guards the cache shared by Flask and the display worker.
_config_lock = threading.Lock()
_config_cache = None

def load_config():
    global _config_cache
    with _config_lock:
        if _config_cache is None:
            if not os.path.exists(config_path):
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                _config_cache = copy.deepcopy(default_config)
            else:
                with open(config_path, 'r') as f:
                    _config_cache = json.load(f)
        return copy.deepcopy(_config_cache)

def save_config_persist(cfg):
    global _config_cache
    with _config_lock:
        _config_cache = copy.deepcopy(cfg)
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

config = load_config()

# ==== E-Paper Setup ====
epd = epd13in3E.EPD()

def epd_init_retry(attempts=3, wait=10):
    """Init the panel, re-initializing SPI/GPIO between attempts.
    Covers slow power-rail settling right after power-on and transient
    glitches; raises after the last attempt so callers can abort."""
    for i in range(1, attempts + 1):
        try:
            epd.Init()
            return
        except Exception as e:
            print("EPD init attempt %d/%d failed: %s" % (i, attempts, e))
            try:
                epd13in3E.epdconfig.module_exit()
            except Exception:
                pass
            if i == attempts:
                raise
            time.sleep(wait)

def get_target_size():
    return (max(epd.width, epd.height), min(epd.width, epd.height))

# ==== Palette ====
_palette_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 0, 0), (255, 255, 255)]
_palette_bytes = sum(_palette_rgb, ()) + (0,) * (768 - len(_palette_rgb) * 3)
_palette_img = Image.new("P", (1, 1))
_palette_img.putpalette(_palette_bytes)

# ==== Dithering Algorithms ====

from lib.dither_core import atkinson_dither as cy_atkinson_dither
from lib.error_dither_core import error_diffuse as cy_error_diffuse

def apply_dithering(image, algorithm):
    """
    Convert a PIL-RGB image into the panel's colors using the selected algorithm.

    floyd-steinberg deliberately dithers in TWO stages: convert("P", ...)
    ignores the custom palette image and dithers to the 216-color web
    palette; epd.getbuffer then re-dithers that to the 6 panel colors.
    The double error diffusion gives a finer, softer look than a single
    direct pass to 6 colors. The Cython algorithms produce exact panel
    colors, which quantize() attaches losslessly and epd.getbuffer maps
    1:1 through its fast LUT path without re-dithering.
    """
    if algorithm == "floyd-steinberg":
        return image.convert("RGB").convert("P", palette=_palette_img, dither=Image.FLOYDSTEINBERG)
    if algorithm == "atkinson":
        return atkinson_dither(image)
    if algorithm == "shiau-fan-2":
        return shiaufan2_dither(image)
    if algorithm == "stucki":
        return stucki_dither(image)
    if algorithm == "burkes":
        return burkes_dither(image)
    # fallback
    return image.convert("RGB").convert("P", palette=_palette_img, dither=Image.FLOYDSTEINBERG)

def atkinson_dither(image: Image.Image) -> Image.Image:
    img = np.array(image.convert("RGB"), dtype=np.float32)
    palette = np.array(_palette_rgb, dtype=np.uint8)
    output = cy_atkinson_dither(img, palette)
    return Image.fromarray(output, mode='RGB').quantize(palette=_palette_img, dither=Image.NONE)

def error_diffusion(image, kernel, divisor, anchor):
    img = np.array(image.convert("RGB"), dtype=np.float32)
    palette = np.array(_palette_rgb, dtype=np.uint8)
    kernel_np = np.array(kernel, dtype=np.int32)
    output = cy_error_diffuse(img, kernel_np, divisor, anchor, palette)
    return Image.fromarray(output, mode='RGB').quantize(palette=_palette_img, dither=Image.NONE)

def shiaufan2_dither(image):
    kernel = [
        [0,0,   0,   8,   4],
        [2,4,   8,   4,   2],
        [1,2,   4,   2,   1]
    ]
    return error_diffusion(image, kernel, divisor=42, anchor=(0,0))

def stucki_dither(image):
    kernel = [
        [0,0,   8,   4,   2],
        [2,4,   8,   4,   2],
        [1,2,   4,   2,   1]
    ]
    return error_diffusion(image, kernel, divisor=42, anchor=(2,0))

def burkes_dither(image):
    kernel = [
        [0,0,   8,   4,   0],
        [2,4,   8,   4,   2]
    ]
    return error_diffusion(image, kernel, divisor=32, anchor=(2,0))

# ==== Image processing ====

def process_image(path_or_file):
    img = Image.open(path_or_file)
    target = get_target_size()
    # JPEG fast path: decode directly at reduced resolution. Multi-MP phone
    # photos otherwise cost ~40MB+ RAM each to decode on a Pi Zero. No-op
    # for other formats. Request the short side so portrait shots (rotated
    # below) still decode large enough for the final resize.
    img.draft("RGB", (min(target), min(target)))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if img.height > img.width:
        img = img.rotate(270, expand=True)

    fit_mode = load_config().get("fit_mode", "pad")

    if fit_mode == "stretch":
        img = img.resize(target, Image.Resampling.LANCZOS)

    elif fit_mode == "zoom":
        img = ImageOps.fit(img, target, method=Image.Resampling.LANCZOS)

    elif fit_mode == "pad":
        # First slightly zoom (e.g. 1.08x) then pad
        zoom_factor = 1.08
        temp_size = (
            int(img.width * zoom_factor),
            int(img.height * zoom_factor)
        )
        img = img.resize(temp_size, Image.Resampling.LANCZOS)
        img = ImageOps.pad(img, target, color="white", method=Image.Resampling.LANCZOS)

    # Optional enhancements
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Color(img).enhance(2.5)
    return img

# ==== Display Worker (single thread + queue) ====
_display_queue = queue.Queue()

current_source_image = None   # original RGB image (before dithering)
current_display_path = None   # file path of the image currently shown
rendered_data = None
rendering_complete = False

def _do_display_update(image):
    """Handles EPD init/display/sleep cycle. Runs in the display worker thread."""
    global current_source_image, rendered_data, rendering_complete
    try:
        # Load latest config to ensure fresh update_count
        cfg = load_config()
        update_count = cfg.get('update_count', 0)

        # Perform full clear on every 12th update
        if update_count % 12 == 0:
            time.sleep(1)
            epd_init_retry()   # Reset + busy-pin wait inside
            time.sleep(1)
            epd.Clear()     # blocks via ReadBusyH until the refresh is done
            time.sleep(1)
            epd.sleep()     # has an internal 2s settle before power-off
            time.sleep(1)
            update_count = 0

        # Update image normally
        time.sleep(1)
        epd_init_retry()
        time.sleep(1)
        try:
            dithered = apply_dithering(image, cfg['dithering'])
            buf = epd.getbuffer(dithered)
            epd.display(buf)
            time.sleep(1)
        finally:
            epd.sleep()
            time.sleep(1)

        # Persist the new update_count on top of a FRESH config: settings
        # changed from the web UI while this render was running must not be
        # clobbered by our stale snapshot
        cfg = load_config()
        cfg['update_count'] = update_count + 1
        save_config_persist(cfg)

        current_source_image = image
        # Build the preview from the actual panel buffer so the browser
        # shows exactly what the e-paper displays
        preview_img = epd.buffer_to_image(buf).rotate(90, expand=True)
        buf_io = io.BytesIO()
        preview_img.save(buf_io, format="PNG")
        rendered_data = base64.b64encode(buf_io.getvalue()).decode('utf-8')
        rendering_complete = True
    except Exception as e:
        print("EPD update error:", e)
        rendering_complete = False

def _do_boot_init():
    """First contact with the panel after power-on. Runs in the display
    worker so the web UI comes up immediately instead of blocking behind
    a slow or unresponsive panel."""
    try:
        time.sleep(3)  # brief power-on settle
        epd_init_retry()
        epd.sleep()
        time.sleep(1)
        print("EPD boot init OK")
    except Exception as e:
        print("EPD boot init error:", e)

def _do_clear():
    """Handles EPD clear cycle. Runs in the display worker thread."""
    try:
        epd_init_retry()
        time.sleep(1)
        epd.Clear()
        time.sleep(1)
    except Exception as e:
        print("EPD clear error:", e)
    finally:
        try:
            epd.sleep()
            time.sleep(1)
        except Exception:
            pass

def _display_worker():
    """Single worker thread that processes display tasks sequentially."""
    while True:
        task = _display_queue.get()
        # Drain queue, keep only the latest task
        while not _display_queue.empty():
            try:
                task = _display_queue.get_nowait()
            except queue.Empty:
                break
        try:
            action = task[0]
            if action == 'display':
                _do_display_update(task[1])
            elif action == 'display_path':
                _do_display_update(process_image(task[1]))
            elif action == 'clear':
                _do_clear()
            elif action == 'boot':
                _do_boot_init()
        except Exception as e:
            print("Display worker error:", e)

threading.Thread(target=_display_worker, daemon=True).start()

def submit_display(image):
    """Submit a pre-processed RGB image for dithering and display."""
    global rendering_complete
    rendering_complete = False
    _display_queue.put(('display', image))

def submit_display_path(path):
    """Submit an image file path for processing, dithering and display."""
    global rendering_complete, current_display_path
    rendering_complete = False
    current_display_path = path
    _display_queue.put(('display_path', path))

def submit_clear():
    """Submit a clear-screen task."""
    _display_queue.put(('clear',))

def initial_display():
    cfg = load_config()
    m = cfg['mode']
    if m == 'single' and cfg['single_image']:
        return
    elif m == 'pool' and cfg['pool_images']:
        fn = random.choice(cfg['pool_images'])
        p = os.path.join(pool_dir, fn)
        if os.path.exists(p):
            submit_display_path(p)

# Panel init happens in the worker; if initial_display() queues a render
# right after, the queue collapses to it and the render does its own init.
_display_queue.put(('boot',))
initial_display()

# ==== Flask & inactivity ====
app = Flask(__name__)
last_activity = time.time()
TIMEOUT = 10*60

@app.before_request
def touch():
    global last_activity
    last_activity = time.time()

def watchdog():
    while True:
        time.sleep(60)
        if time.time()-last_activity>TIMEOUT:
            GPIO.output(16, GPIO.HIGH)
            time.sleep(15)
            os.system("sudo shutdown now")
            break

threading.Thread(target=watchdog, daemon=True).start()

# ==== Web UI Template ====

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Spectra 6 Frame</title>
  <style>
    :root{
      --bg:#10141a; --card:#1a212b; --line:#2a3442; --text:#e8edf4;
      --muted:#8a97a8; --accent:#4f9cf9; --green:#3ecf8e; --red:#ef6363;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
    .wrap{max-width:980px;margin:0 auto;padding:24px 16px 64px;}
    header{display:flex;align-items:center;gap:12px;margin-bottom:20px;flex-wrap:wrap;}
    header h1{font-size:21px;margin:0;font-weight:650;}
    .badge{font-size:12px;color:var(--muted);border:1px solid var(--line);
           border-radius:999px;padding:3px 10px;}
    #msg{margin-left:auto;}
    .status{display:inline-flex;align-items:center;gap:8px;font-size:13px;
            padding:6px 12px;border-radius:999px;border:1px solid var(--line);}
    .status.info{color:var(--accent)} .status.success{color:var(--green)}
    .status.danger{color:var(--red)}
    .spin{width:12px;height:12px;border:2px solid currentColor;flex:none;
          border-top-color:transparent;border-radius:50%;animation:sp 1s linear infinite;}
    @keyframes sp{to{transform:rotate(360deg)}}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    @media(max-width:760px){.grid{grid-template-columns:1fr}}
    .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;}
    .card h2{font-size:13px;margin:0 0 12px;font-weight:600;color:var(--muted);
             text-transform:uppercase;letter-spacing:.07em;}
    .span2{grid-column:1/-1;}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px;}
    .hint{font-size:12px;color:var(--muted);margin-top:8px;}
    .btn{background:#243043;color:var(--text);border:1px solid var(--line);
         border-radius:9px;padding:8px 14px;font-size:14px;cursor:pointer;}
    .btn:hover{background:#2c3b52}
    .btn.primary{background:var(--accent);border-color:transparent;color:#fff;}
    .btn.primary:hover{filter:brightness(1.12)}
    .btn.danger{background:transparent;border-color:#5c3434;color:var(--red);}
    .btn.danger:hover{background:#3a2222}
    select{background:#0f1620;color:var(--text);border:1px solid var(--line);
           border-radius:9px;padding:8px 10px;font-size:14px;min-width:190px;}
    label.fld{font-size:13px;color:var(--muted);display:block;margin-bottom:4px;}
    input[type=file]{color:var(--muted);font-size:13px;max-width:100%;}
    input[type=file]::file-selector-button{
      background:#243043;color:var(--text);border:1px solid var(--line);
      border-radius:9px;padding:7px 12px;font-size:13px;cursor:pointer;margin-right:10px;}
    .preview-box{background:#0b0f14;border:1px solid var(--line);border-radius:10px;
                 display:flex;align-items:center;justify-content:center;
                 min-height:200px;overflow:hidden;}
    .preview-box img{max-width:100%;height:auto;display:none;}
    .preview-box .ph{color:var(--muted);font-size:13px;}
    .pool{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));
          gap:10px;margin-top:14px;}
    .pi{position:relative;border:2px solid var(--line);border-radius:10px;
        overflow:hidden;background:#0b0f14;}
    .pi.current{border-color:var(--green);}
    .pi.current::after{content:"on display";position:absolute;left:6px;top:6px;
        background:var(--green);color:#06291a;font-size:10px;font-weight:700;
        padding:2px 6px;border-radius:6px;pointer-events:none;}
    .pi img{width:100%;height:96px;object-fit:cover;display:block;cursor:pointer;}
    .pi img:hover{opacity:.8}
    .pi .nm{font-size:11px;color:var(--muted);padding:5px 8px;white-space:nowrap;
            overflow:hidden;text-overflow:ellipsis;}
    .pi .rm{position:absolute;top:5px;right:5px;width:22px;height:22px;border-radius:50%;
            border:none;background:rgba(0,0,0,.55);color:#fff;cursor:pointer;
            font-size:12px;line-height:1;}
    .pi .rm:hover{background:var(--red);}
    details{margin-top:16px;}
    summary{cursor:pointer;color:var(--muted);font-size:13px;}
    pre{background:#0b0f14;border:1px solid var(--line);border-radius:10px;
        padding:12px;font-size:12px;color:var(--muted);overflow:auto;}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Spectra 6 Frame</h1>
    <span class="badge" id="modeBadge">mode: &hellip;</span>
    <div id="msg"></div>
  </header>

  <div class="grid">
    <div class="card span2">
      <h2>Now on display</h2>
      <div class="preview-box">
        <div class="ph" id="previewPh">No render yet</div>
        <img id="preview" alt="Panel preview">
      </div>
      <div class="row">
        <button id="rotateBtn" class="btn">Rotate 90&deg;</button>
        <button id="clearBtn" class="btn danger">Clear e-paper</button>
      </div>
      <div class="hint">Preview is decoded from the exact buffer sent to the panel.</div>
    </div>

    <div class="card">
      <h2>Show a single image</h2>
      <form id="singleForm">
        <input type="file" name="image" accept="image/*" required>
        <div class="row">
          <button class="btn primary">Upload &amp; show</button>
        </div>
      </form>
    </div>

    <div class="card">
      <h2>Settings</h2>
      <label class="fld" for="fitModeSelect">Image fit</label>
      <select id="fitModeSelect">
        <option value="pad">Pad with white borders</option>
        <option value="zoom">Zoom to fill</option>
        <option value="stretch">Stretch</option>
      </select>
      <label class="fld" for="ditherSelect" style="margin-top:10px;">Dithering</label>
      <select id="ditherSelect">
        <option value="floyd-steinberg">Floyd&ndash;Steinberg</option>
        <option value="atkinson">Atkinson</option>
        <option value="shiau-fan-2">Shiau-Fan 2</option>
        <option value="stucki">Stucki</option>
        <option value="burkes">Burkes</option>
      </select>
      <div class="row">
        <button id="applySettings" class="btn primary">Apply &amp; re-render</button>
      </div>
    </div>

    <div class="card span2">
      <h2>Image pool (rotation)</h2>
      <form id="addPoolForm">
        <input type="file" name="images" accept="image/*" multiple>
        <div class="row">
          <button class="btn">Add to pool</button>
          <button type="button" id="setPool" class="btn primary">Use pool mode</button>
        </div>
      </form>
      <div class="hint">A random pool image is shown on every wake-up. Click a thumbnail to show it now.</div>
      <div class="hint" id="poolEmpty">Pool is empty.</div>
      <div class="pool" id="poolGrid"></div>
    </div>
  </div>

  <details>
    <summary>Raw configuration</summary>
    <pre id="configDisplay"></pre>
  </details>
</div>

<script>
const $ = id => document.getElementById(id);

function showMsg(txt, cls = "info", busy = false) {
  $('msg').innerHTML = `<span class="status ${cls}">${busy ? '<span class="spin"></span>' : ''}${txt}</span>`;
}

// Single polling chain: pressing several buttons must not stack up
// parallel /preview pollers.
let pollTimer = null;
function pollPreview() {
  clearTimeout(pollTimer);
  fetch('/preview')
    .then(r => r.json())
    .then(data => {
      if (data.rendered_image) {
        $('preview').src = "data:image/png;base64," + data.rendered_image;
        $('preview').style.display = 'block';
        $('previewPh').style.display = 'none';
        showMsg("Up to date", "success");
        loadConfig();
      } else {
        pollTimer = setTimeout(pollPreview, 3000);
      }
    })
    .catch(() => { pollTimer = setTimeout(pollPreview, 5000); });
}

function doAction(url, opts, msg) {
  showMsg(msg, "info", true);
  fetch(url, opts).then(r => {
    if (r.ok) {
      showMsg("Rendering&hellip; e-paper refresh takes a minute", "info", true);
      pollPreview();
    } else {
      return r.json().then(j => showMsg(j.error || ("Error " + r.status), "danger"))
                     .catch(() => showMsg("Error " + r.status, "danger"));
    }
  }).catch(e => showMsg("Request failed: " + e, "danger"));
}

function loadConfig() {
  fetch('/config').then(r => r.json()).then(c => {
    $('configDisplay').textContent = JSON.stringify(c, null, 2);
    if (c.fit_mode) $('fitModeSelect').value = c.fit_mode;
    if (c.dithering) $('ditherSelect').value = c.dithering;
    $('modeBadge').textContent = "mode: " + c.mode;
  }).catch(() => {});

  fetch('/pool/list').then(r => r.json()).then(p => {
    const grid = $('poolGrid');
    grid.innerHTML = "";
    const images = p.images || [];
    $('poolEmpty').style.display = images.length ? 'none' : 'block';
    images.forEach(fn => {
      const card = document.createElement('div');
      card.className = 'pi' + (fn === p.current ? ' current' : '');

      const img = document.createElement('img');
      img.src = '/pool/thumb/' + encodeURIComponent(fn);
      img.title = "Show on frame";
      img.onclick = () => doAction('/mode/pool/show', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: fn})
      }, "Showing " + fn + "&hellip;");

      const nm = document.createElement('div');
      nm.className = 'nm'; nm.textContent = fn; nm.title = fn;

      const rm = document.createElement('button');
      rm.className = 'rm'; rm.textContent = '✕'; rm.title = "Remove from pool";
      rm.onclick = ev => {
        ev.stopPropagation();
        fetch('/mode/pool/remove', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({filename: fn})
        }).then(loadConfig)
          .catch(e => showMsg("Request failed: " + e, "danger"));
      };

      card.append(img, nm, rm);
      grid.appendChild(card);
    });
  }).catch(() => {});
}

$('singleForm').onsubmit = e => {
  e.preventDefault();
  doAction('/mode/single', {method:'POST', body:new FormData(e.target)}, "Uploading&hellip;");
};

$('addPoolForm').onsubmit = e => {
  e.preventDefault();
  if (!e.target.images.files.length) { showMsg("Choose files first", "danger"); return; }
  showMsg("Adding to pool&hellip;", "info", true);
  fetch('/mode/pool/add', {method:'POST', body:new FormData(e.target)})
    .then(r => {
      if (r.ok) { showMsg("Added to pool", "success"); e.target.reset(); loadConfig(); }
      else showMsg("Upload failed (" + r.status + ")", "danger");
    })
    .catch(e => showMsg("Request failed: " + e, "danger"));
};

$('setPool').onclick = () =>
  doAction('/mode/pool/set', {method:'POST'}, "Setting pool mode&hellip;");

$('applySettings').onclick = () => {
  // sequential: parallel requests would race on the config read-modify-write
  showMsg("Applying settings&hellip;", "info", true);
  fetch('/mode/dither/set', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({algorithm: $('ditherSelect').value})
  }).then(r => {
    if (!r.ok) throw new Error("dither: HTTP " + r.status);
    doAction('/mode/fit/set', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({fit_mode: $('fitModeSelect').value})
    }, "Applying settings&hellip;");
  }).catch(e => showMsg("Failed: " + e.message, "danger"));
};

$('clearBtn').onclick = () => {
  showMsg("Clearing screen&hellip;", "info", true);
  fetch('/clear', {method:'POST'})
    .then(r => showMsg(r.ok ? "Clear queued &mdash; panel will blank shortly" : "Error " + r.status,
                       r.ok ? "success" : "danger"))
    .catch(e => showMsg("Request failed: " + e, "danger"));
};

$('rotateBtn').onclick = () =>
  doAction('/rotate', {method:'GET'}, "Rotating&hellip;");

loadConfig();
pollPreview();
</script>
</body>
</html>
"""

# ==== Flask Endpoints ====

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/preview', methods=['GET'])
def preview():
    # Only expose the image once the current render is done, otherwise the
    # UI instantly shows the previous (stale) render after every button press
    return jsonify(rendered_image=rendered_data if rendering_complete else None,
                   success=rendering_complete)

@app.route('/mode/single', methods=['POST'])
def set_single():
    f = request.files.get('image')
    if not f:
        return jsonify(error="No file"), 400
    fn = secure_filename(f.filename)
    # clear old single image
    for e in os.listdir(single_dir):
        os.remove(os.path.join(single_dir, e))
    path = os.path.join(single_dir, fn)
    f.save(path)
    cfg = load_config()  # load fresh config
    cfg['mode'] = 'single'
    cfg['single_image'] = fn
    save_config_persist(cfg)  # save immediately
    submit_display_path(path)
    return jsonify(success=True)

@app.route('/mode/pool/add', methods=['POST'])
def pool_add():
    files = request.files.getlist('images')
    cfg = load_config()
    for f in files:
        fn = secure_filename(f.filename)
        if not fn:
            continue
        path = os.path.join(pool_dir, fn)
        f.save(path)
        if fn not in cfg['pool_images']:
            cfg['pool_images'].append(fn)
    save_config_persist(cfg)
    return jsonify(success=True)

@app.route('/pool/list', methods=['GET'])
def pool_list():
    cfg = load_config()
    current = os.path.basename(current_display_path) if current_display_path else None
    return jsonify(images=cfg.get('pool_images', []), current=current)

def _thumb_path(fn):
    # extension appended (not replaced) so distinct uploads never collide
    return os.path.join(thumbs_dir, fn + '.jpg')

# One thumbnail generation at a time: the browser requests all of them in
# parallel, and decoding several multi-MP photos at once would exhaust the
# Pi Zero's RAM
_thumb_lock = threading.Lock()

@app.route('/pool/thumb/<path:filename>', methods=['GET'])
def pool_thumb(filename):
    fn = secure_filename(filename)
    src = os.path.join(pool_dir, fn)
    if not fn or not os.path.exists(src):
        return jsonify(error="Not found"), 404
    th = _thumb_path(fn)
    try:
        with _thumb_lock:
            if not os.path.exists(th) or os.path.getmtime(th) < os.path.getmtime(src):
                img = Image.open(src)
                # decode JPEGs at ~1/8 resolution instead of full size
                img.draft("RGB", (480, 360))
                img.thumbnail((240, 180))
                img = ImageOps.exif_transpose(img)
                img.convert("RGB").save(th, "JPEG", quality=80)
    except Exception as e:
        return jsonify(error=str(e)), 500
    return send_file(th, mimetype='image/jpeg', max_age=3600, conditional=True)

@app.route('/mode/pool/show', methods=['POST'])
def pool_show():
    data = request.get_json() or {}
    fn = data.get('filename')
    cfg = load_config()
    if fn not in cfg.get('pool_images', []):
        return jsonify(error="Not in pool"), 404
    p = os.path.join(pool_dir, fn)
    if not os.path.exists(p):
        return jsonify(error="File missing"), 404
    submit_display_path(p)
    return jsonify(success=True)

@app.route('/mode/pool/remove', methods=['POST'])
def pool_remove():
    data = request.get_json() or {}
    fn = data.get('filename')
    cfg = load_config()
    if fn in cfg['pool_images']:
        cfg['pool_images'].remove(fn)
        save_config_persist(cfg)
        for p in (os.path.join(pool_dir, fn), _thumb_path(fn)):
            if os.path.exists(p):
                os.remove(p)
    return jsonify(success=True)

@app.route('/mode/pool/set', methods=['POST'])
def set_pool_mode():
    cfg = load_config()
    if not cfg['pool_images']:
        return jsonify(error="No images in pool"), 400
    cfg['mode'] = 'pool'
    save_config_persist(cfg)
    fn = random.choice(cfg['pool_images'])
    path = os.path.join(pool_dir, fn)
    submit_display_path(path)
    return jsonify(success=True)

@app.route('/mode/art/set', methods=['POST'])
def set_art_mode():
    cfg = load_config()
    cfg['mode'] = 'art'
    save_config_persist(cfg)
    return jsonify(success=True)

@app.route('/mode/fit/set', methods=['POST'])
def set_fit_mode():
    data = request.get_json() or {}
    mode = data.get("fit_mode", "pad")
    if mode not in ("pad", "zoom", "stretch"):
        return jsonify(error="Invalid fit mode"), 400

    cfg = load_config()
    cfg['fit_mode'] = mode
    save_config_persist(cfg)

    # Reprocess and re-render current image from disk
    try:
        if cfg['mode'] == 'single' and cfg['single_image']:
            path = os.path.join(single_dir, cfg['single_image'])
        elif cfg['mode'] == 'pool' and cfg['pool_images']:
            # Re-render the image currently on screen, not just the first one
            path = current_display_path
            if not path or not os.path.exists(path):
                path = os.path.join(pool_dir, cfg['pool_images'][0])
        else:
            return jsonify(success=True)  # nothing to render

        submit_display_path(path)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/mode/dither/set', methods=['POST'])
def set_dither():
    data = request.get_json() or {}
    alg  = data.get('algorithm')
    if alg in ("floyd-steinberg","atkinson","shiau-fan-2","stucki","burkes"):
        cfg = load_config()
        cfg['dithering'] = alg
        save_config_persist(cfg)
        # re-render current image with new dithering
        if current_source_image is not None:
            submit_display(current_source_image)
        else:
            # No in-memory image yet (e.g. right after boot): re-render from disk
            if cfg['mode'] == 'single' and cfg['single_image']:
                p = os.path.join(single_dir, cfg['single_image'])
            elif cfg['mode'] == 'pool' and cfg['pool_images']:
                p = current_display_path or os.path.join(pool_dir, cfg['pool_images'][0])
            else:
                p = None
            if p and os.path.exists(p):
                submit_display_path(p)
        return jsonify(success=True)
    return jsonify(error="Invalid algorithm"), 400

@app.route('/rotate', methods=['GET'])
def rotate():
    global rendering_complete
    if current_source_image is None:
        return jsonify(error="No image to rotate"), 400
    try:
        rot = current_source_image.rotate(90, expand=True)
        rot = ImageOps.fit(rot, get_target_size(), method=Image.Resampling.LANCZOS)
        rendering_complete = False
        submit_display(rot)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/config', methods=['GET'])
def get_config():
    # return the live config object
    cfg = load_config()
    return jsonify(cfg)

@app.route('/clear', methods=['POST'])
def clear_endpoint():
    submit_clear()
    return jsonify(success=True)


if __name__ == '__main__':
    # threaded=True keeps the UI responsive while the display worker is busy
    app.run(host='0.0.0.0', port=80, threaded=True)
