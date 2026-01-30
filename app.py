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
import numpy as np
import RPi.GPIO as GPIO

from flask import Flask, request, render_template_string, jsonify
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


os.makedirs(single_dir, exist_ok=True)
os.makedirs(pool_dir, exist_ok=True)

default_config = {
    "mode":          "single",
    "single_image":  "",
    "pool_images":   [],
    "dithering":     "floyd-steinberg",
    "update_count":  0,
    "fit_mode":      "pad"  # pad | zoom | stretch
}

def load_config():
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config.copy()
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config_persist(cfg):
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

config = load_config()

# ==== E-Paper Setup ====
epd = epd13in3E.EPD()
try:
    time.sleep(5)  # brief settle after power-on (launcher.sh already waits 15s)
    epd.Init()
#    epd.Clear()
    epd.sleep()
    time.sleep(1)
except Exception as e:
    print("EPD init error:", e)

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
    Convert a PIL-RGB image into our 6-color palette using the selected algorithm.
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
    return Image.fromarray(output, mode='RGB').convert("P", palette=_palette_img, dither=Image.NONE)

def error_diffusion(image, kernel, divisor, anchor):
    img = np.array(image.convert("RGB"), dtype=np.float32)
    palette = np.array(_palette_rgb, dtype=np.uint8)
    kernel_np = np.array(kernel, dtype=np.int32)
    output = cy_error_diffuse(img, kernel_np, divisor, anchor, palette)
    return Image.fromarray(output, mode='RGB').convert("P", palette=_palette_img, dither=Image.NONE)

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
    img = Image.open(path_or_file).convert("RGB")
    img = ImageOps.exif_transpose(img)
    if img.height > img.width:
        img = img.rotate(270, expand=True)

    fit_mode = load_config().get("fit_mode", "pad")
    target = get_target_size()

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
            time.sleep(0.5)
            epd.Init()     # waits for busy internally
            epd.Clear()    # waits for busy via TurnOnDisplay → ReadBusyH
            time.sleep(0.2)
            epd.sleep()    # has internal 2s delay
            time.sleep(0.5)
            update_count = 0

        # Update image normally
        time.sleep(0.5)
        epd.Init()         # waits for busy internally
        try:
            dithered = apply_dithering(image, cfg['dithering'])
            buf = epd.getbuffer(dithered)
            epd.display(buf)  # waits for busy via TurnOnDisplay → ReadBusyH
            time.sleep(0.2)
        finally:
            epd.sleep()    # has internal 2s delay

        # Update and save config with incremented update_count
        cfg['update_count'] = update_count + 1
        save_config_persist(cfg)

        current_source_image = image
        buf_io = io.BytesIO()
        dithered.rotate(180).save(buf_io, format="PNG")
        rendered_data = base64.b64encode(buf_io.getvalue()).decode('utf-8')
        rendering_complete = True
    except Exception as e:
        print("EPD update error:", e)
        rendering_complete = False

def _do_clear():
    """Handles EPD clear cycle. Runs in the display worker thread."""
    try:
        epd.Init()
        epd.Clear()
        time.sleep(0.2)
    except Exception as e:
        print("EPD clear error:", e)
    finally:
        try:
            epd.sleep()
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
    global rendering_complete
    rendering_complete = False
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
  <title>Spectra 6 Picture Frame</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <style>body{padding:20px;} .section{margin-bottom:30px;}</style>
</head>
<body>
  <h1 class="mb-4">Spectra 6 Picture Frame</h1>

  <div class="section">
    <h3>Single Image Mode</h3>
    <form id="singleForm" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button class="btn btn-primary mt-2">Upload & Set</button>
    </form>
  </div>

  <div class="section">
    <h3>Image Pool Mode</h3>
    <form id="addPoolForm" enctype="multipart/form-data">
      <input type="file" name="images" accept="image/*" multiple>
      <button class="btn btn-secondary mt-2">Add to Pool</button>
    </form>
    <button id="setPool" class="btn btn-primary mt-2">Use Pool Mode</button>
    <h5 class="mt-3">Current Pool</h5>
    <ul id="poolList"></ul>
  </div>
  <div class="section">
    <h3>Image Fit Mode</h3>
      <select id="fitModeSelect" class="form-control" style="max-width:300px;">
        <option value="pad">Pad with White Borders</option>
        <option value="zoom">Zoom to Fill</option>
        <option value="stretch">Stretch</option>
      </select>
    <button id="setFitMode" class="btn btn-primary mt-2">Apply & Refresh</button>
  </div>
  <div class="section">
    <h3>Dithering Algorithm</h3>
    <select id="ditherSelect" class="form-control" style="max-width:300px;">
      <option value="floyd-steinberg">Floyd–Steinberg</option>
      <option value="atkinson">Atkinson</option>
      <option value="shiau-fan-2">Shiau-Fan 2</option>
      <option value="stucki">Stucki</option>
      <option value="burkes">Burkes</option>
    </select>
    <button id="setDither" class="btn btn-primary mt-2">Apply & Refresh</button>
  </div>

  <div class="section">
    <h3>Art of the Day</h3>
    <button id="setArt" class="btn btn-primary">Set Art Mode</button>
    <p class="text-muted">(Not implemented)</p>
  </div>

  <div class="section">
    <h3>Clear Screen</h3>
    <button id="clearBtn" class="btn btn-danger">Clear E-Paper</button>
  </div>

  <div class="section">
    <h3>Live Preview</h3>
    <div id="msg"></div>
    <img id="preview" src="" class="img-fluid" style="max-width:400px;">
    <button id="rotateBtn" class="btn btn-warning mt-2">Rotate 90°</button>
  </div>
  <div class="section">
    <h3>Current Configuration</h3>
    <pre id="configDisplay" class="bg-light p-3"></pre>
  </div>
<script>
function showMsg(txt, cls="info") {
  document.getElementById('msg').innerHTML = `<div class="alert alert-${cls}">${txt}</div>`;
}

function pollPreview() {
  fetch('/preview')
    .then(response => response.json())
    .then(data => {
      if (data.rendered_image) {
        document.getElementById('preview').src = "data:image/png;base64," + data.rendered_image;
        showMsg("Rendering complete!", "success");
      } else {
        setTimeout(pollPreview, 3000);
      }
    })
    .catch(e => showMsg("Error fetching preview: " + e, "danger"));
}

function loadConfig(){
  fetch('/config').then(r=>r.json()).then(c=>{
    document.getElementById('configDisplay').innerText =
      JSON.stringify(c, null, 2);

    if (c.fit_mode) {
      document.getElementById('fitModeSelect').value = c.fit_mode;
    }
    if (c.dithering) {
      document.getElementById('ditherSelect').value = c.dithering;
    }
  });

  fetch('/pool/list').then(r=>r.json()).then(p=>{
    let ul = document.getElementById('poolList');
    ul.innerHTML = "";
    p.forEach(fn => {
      const li = document.createElement('li');
      li.textContent = fn + " ";
      const btn = document.createElement('button');
      btn.textContent = "Remove";
      btn.className = "btn btn-sm btn-danger ml-2";
      btn.onclick = () => {
        fetch('/mode/pool/remove', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({filename: fn})
        }).then(loadPool);
      };
      li.appendChild(btn);
      ul.appendChild(li);
    });
  });
}

document.getElementById('singleForm').onsubmit = e => {
  e.preventDefault();
  showMsg("Uploading...", "info");
  fetch('/mode/single', {method:'POST', body:new FormData(e.target)})
    .then(()=>{ showMsg("Rendering...", "info"); pollPreview(); });
};

document.getElementById('addPoolForm').onsubmit = e => {
  e.preventDefault();
  showMsg("Adding to pool...", "info");
  fetch('/mode/pool/add', {method:'POST', body:new FormData(e.target)})
    .then(()=>{ showMsg("Done.", "success"); loadConfig(); });
};

document.getElementById('setPool').onclick = () => {
  showMsg("Setting pool mode...", "info");
  fetch('/mode/pool/set', {method:'POST'})
    .then(()=>{ showMsg("Rendering...", "info"); pollPreview(); });
};

document.getElementById('setFitMode').onclick = () => {
  const mode = document.getElementById('fitModeSelect').value;
  showMsg("Applying fit mode...", "info");
  fetch('/mode/fit/set', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({fit_mode: mode})
  }).then(()=> { showMsg("Rendering...", "info"); pollPreview(); });
};

document.getElementById('setDither').onclick = () => {
  const alg = document.getElementById('ditherSelect').value;
  showMsg("Applying dithering...", "info");
  fetch('/mode/dither/set', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({algorithm: alg})
  }).then(()=>{ showMsg("Rendering...", "info"); pollPreview(); });
};

document.getElementById('clearBtn').onclick = () => {
  showMsg("Clearing screen...", "info");
  fetch('/clear', {method:'POST'})
    .then(()=> showMsg("Screen cleared", "success"));
};

document.getElementById('rotateBtn').onclick = () => {
  showMsg("Rotating...", "info");
  fetch('/rotate', {method:'GET'}).then(()=>{ showMsg("Rendering...", "info"); pollPreview(); });
};

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
    return jsonify(rendered_image=rendered_data, success=rendering_complete)

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
        path = os.path.join(pool_dir, fn)
        f.save(path)
        if fn not in cfg['pool_images']:
            cfg['pool_images'].append(fn)
    save_config_persist(cfg)
    return jsonify(success=True)

@app.route('/pool/list', methods=['GET'])
def pool_list():
    return jsonify(load_config().get('pool_images', []))

@app.route('/mode/pool/remove', methods=['POST'])
def pool_remove():
    data = request.get_json() or {}
    fn = data.get('filename')
    cfg = load_config()
    if fn in cfg['pool_images']:
        cfg['pool_images'].remove(fn)
        save_config_persist(cfg)
        p = os.path.join(pool_dir, fn)
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
    if alg in ("floyd-steinberg","atkinson","shiau-fan-2","jarvis-judice-ninke","stucki","burkes"):
        cfg = load_config()
        cfg['dithering'] = alg
        save_config_persist(cfg)
        # re-render current image with new dithering
        if current_source_image is not None:
            submit_display(current_source_image)
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
    app.run(host='0.0.0.0', port=80)
