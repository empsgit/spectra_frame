#!/usr/bin/env python3

import sys
import os
import time
import threading
import io
import base64
import json
import random
import numpy as np


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

os.makedirs(single_dir, exist_ok=True)
os.makedirs(pool_dir, exist_ok=True)

default_config = {
    "mode":          "single",
    "single_image":  "",
    "pool_images":   [],
    "dithering":     "floyd-steinberg",
    "fit_mode":      "pad"  # pad | zoom | stretch
}

def load_config():
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config.copy()
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config():
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

config = load_config()

# ==== E-Paper Setup ====
epd = epd13in3E.EPD()
try:
    epd.Init()
#    epd.Clear()
    epd.sleep()
except Exception as e:
    print("EPD init error:", e)

def get_target_size():
    return (max(epd.width, epd.height), min(epd.width, epd.height))

# prepare palette
_custom_palette = [
    255,0,0,    # red
    0,255,0,    # green
    0,0,255,    # blue
    255,255,0,  # yellow
    0,0,0,      # black
    255,255,255 # white
] + [0] * (768 - 6*3)
_palette_img = Image.new("P",(1,1))
_palette_img.putpalette(_custom_palette)

# ==== Dithering Algorithms ====
def atkinson_dither_fast(image):
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    arr = arr.astype(np.int32)

    palette = np.array([
        [255,0,0],    # red
        [0,255,0],    # green
        [0,0,255],    # blue
        [255,255,0],  # yellow
        [0,0,0],      # black
        [255,255,255] # white
    ])

    def find_closest(color):
        diff = palette - color
        dist = np.sum(diff**2, axis=1)
        return palette[np.argmin(dist)]

    for y in range(h):
        for x in range(w):
            old_pixel = arr[y, x]
            new_pixel = find_closest(old_pixel)
            arr[y, x] = new_pixel
            error = (old_pixel - new_pixel) // 8
            for dx, dy in [(1,0),(2,0),(-1,1),(0,1),(1,1),(0,2)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    arr[ny, nx] = np.clip(arr[ny, nx] + error, 0, 255)

    dithered_img = Image.fromarray(arr.astype(np.uint8), "RGB")
    return dithered_img.convert("P", palette=_palette_img, dither=Image.NONE)





def apply_dithering(image, algorithm):
    """
    Convert a PIL-RGB image into our 6-color palette using the selected algorithm.
    """
    if algorithm == "floyd-steinberg":
        return image.convert("RGB").convert("P", palette=_palette_img, dither=Image.FLOYDSTEINBERG)
    if algorithm == "atkinson":
        return atkinson_dither_fast(image)
    if algorithm == "shiau-fan-2":
        return shiaufan2_dither(image)
#    if algorithm == "jarvis-judice-ninke":
#        return jarvis_judice_ninke_dither(image)
    if algorithm == "stucki":
        return stucki_dither(image)
    if algorithm == "burkes":
        return burkes_dither(image)
    # fallback
    return image.convert("RGB").convert("P", palette=_palette_img, dither=Image.FLOYDSTEINBERG)

def atkinson_dither(image):
    img = image.convert("RGB")
    pixels = img.load()
    w, h = img.size
    palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,0,0),(255,255,255)]
    for y in range(h):
        for x in range(w):
            old = pixels[x,y]
            new = min(palette, key=lambda c: sum((old[i]-c[i])**2 for i in range(3)))
            pixels[x,y] = new
            err = tuple(old[i]-new[i] for i in range(3))
            for dx,dy in [(1,0),(2,0),(-1,1),(0,1),(1,1),(0,2)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h:
                    r,g,b = pixels[nx,ny]
                    pixels[nx,ny] = (
                        max(0, min(255, r + err[0]//8)),
                        max(0, min(255, g + err[1]//8)),
                        max(0, min(255, b + err[2]//8))
                    )
    return img.convert("P", palette=_palette_img, dither=Image.NONE)

def error_diffusion(image, kernel, divisor, anchor):
    img = image.convert("RGB")
    pixels = img.load()
    w, h = img.size
    palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,0,0),(255,255,255)]
    kh, kw = len(kernel), len(kernel[0])
    ax, ay = anchor
    for y in range(h):
        for x in range(w):
            old = pixels[x,y]
            new = min(palette, key=lambda c: sum((old[i]-c[i])**2 for i in range(3)))
            pixels[x,y] = new
            err = tuple(old[i]-new[i] for i in range(3))
            for dy in range(kh):
                for dx in range(kw):
                    val = kernel[dy][dx]
                    if val == 0: continue
                    nx, ny = x + dx - ax, y + dy - ay
                    if 0 <= nx < w and 0 <= ny < h:
                        r,g,b = pixels[nx,ny]
                        pixels[nx,ny] = (
                            max(0, min(255, r + err[0]*val//divisor)),
                            max(0, min(255, g + err[1]*val//divisor)),
                            max(0, min(255, b + err[2]*val//divisor))
                        )
    return img.convert("P", palette=_palette_img, dither=Image.NONE)

def shiaufan2_dither(image):
    kernel = [
        [0,0,   0,   8,   4],
        [2,4,   8,   4,   2],
        [1,2,   4,   2,   1]
    ]
    return error_diffusion(image, kernel, divisor=42, anchor=(0,0))

#def jarvis_judice_ninke_dither(image):
#    kernel = [
#        [0,0,   7,   5,   3],
#        [3,5,   7,   5,   3],
#        [1,3,   5,   3,   1]
#    ]
#    return error_diffusion(image, kernel, divisor=48, anchor=(2,0))

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

# ==== Image process & display ====
update_counter = 0
counter_lock = threading.Lock()

def display_image(image):
    global update_counter
    with counter_lock:
        update_counter += 1
        count = update_counter

    # On every 10th update, do a full clear first
    if count % 5 == 0:
        epd.Init()
        epd.Clear()
        epd.sleep()

    # Standard update
    epd.Init()
    img = ImageOps.fit(image, get_target_size(), method=Image.Resampling.LANCZOS)
    dithered = apply_dithering(img, config['dithering'])
    buf = epd.getbuffer(dithered)
    epd.display(buf)
    epd.sleep()

    return dithered

def clear_screen():
    """Force a full clear + sleep in a background thread."""
    def _clear():
        epd.Init()
        epd.Clear()
        epd.sleep()
    threading.Thread(target=_clear, daemon=True).start()

def process_image(path_or_file):
    img = Image.open(path_or_file).convert("RGB")
    img = ImageOps.exif_transpose(img)
    if img.height > img.width:
        img = img.rotate(90, expand=True)
    img = img.rotate(180, expand=True)

    fit_mode = config.get("fit_mode", "pad")
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
    img = ImageEnhance.Color(img).enhance(3.0)
    return img

current_image = None
rendered_data = None
rendering_complete = False

def update_epaper_thread(image):
    global current_image, rendered_data, rendering_complete
    try:
        d = display_image(image)
        current_image = d
        buf = io.BytesIO()
        d.rotate(180).save(buf, format="PNG")
        rendered_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        rendering_complete = True
    except Exception as e:
        print("EPD update error:", e)
        rendering_complete = False

def initial_display():
    m = config['mode']
    if m == 'single' and config['single_image']:
        p = os.path.join(single_dir, config['single_image'])
        if os.path.exists(p):
            update_epaper_thread(process_image(p))
    elif m == 'pool' and config['pool_images']:
        fn = random.choice(config['pool_images'])
        p = os.path.join(pool_dir, fn)
        if os.path.exists(p):
            update_epaper_thread(process_image(p))

threading.Thread(target=initial_display, daemon=True).start()

# ==== Flask & inactivity ====
app = Flask(__name__)
last_activity = time.time()
TIMEOUT = 20*60

@app.before_request
def touch():
    global last_activity
    last_activity = time.time()

def watchdog():
    while True:
        time.sleep(60)
        if time.time()-last_activity>TIMEOUT:
            os.system("sudo restart now")
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
      <option value="jarvis-judice-ninke">Jarvis–Judice–Ninke</option>
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

document.getElementById('setFitMode').onclick = () => {
  const mode = document.getElementById('fitModeSelect').value;
  showMsg("Applying fit mode...", "info");
  fetch('/mode/fit/set', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({fit_mode: mode})
  }).then(()=> {
    showMsg("Rendering...", "info");
    pollPreview();
  });
};

function showMsg(txt, cls="info") {
  document.getElementById('msg').innerHTML = `<div class="alert alert-${cls}">${txt}</div>`;
}

  function loadConfig(){
  fetch('/config').then(r=>r.json()).then(c=>{
    document.getElementById('configDisplay').innerText =
      JSON.stringify(c, null, 2);

    // Sync dropdowns to config values
    if (c.fit_mode) {
      document.getElementById('fitModeSelect').value = c.fit_mode;
    }
    if (c.dithering) {
      document.getElementById('ditherSelect').value = c.dithering;
    }
  });

  fetch('/pool/list').then(r=>r.json()).then(p=>{
    let ul = document.getElementById('poolList');
    ul.innerHTML = p.map(i=>`<li>${i}</li>`).join('');
  });
}

function loadPool() {
  fetch('/pool/list').then(r=>r.json()).then(arr=>{
    const ul = document.getElementById('poolList');
    ul.innerHTML = "";
    arr.forEach(fn => {
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

// Handlers
document.getElementById('singleForm').onsubmit = e => {
  e.preventDefault();
  showMsg("Uploading...", "info");
  fetch('/mode/single', {method:'POST', body:new FormData(e.target)})
    .then(()=>{
      showMsg("Rendering...", "info");
      pollPreview();
    });
};

document.getElementById('addPoolForm').onsubmit = e => {
  e.preventDefault();
  showMsg("Adding to pool...", "info");
  fetch('/mode/pool/add', {method:'POST', body:new FormData(e.target)})
    .then(()=>{
      showMsg("Done.", "success");
      loadPool();
    });
};

document.getElementById('setPool').onclick = () => {
  showMsg("Setting pool mode...", "info");
  fetch('/mode/pool/set', {method:'POST'})
    .then(()=>{
      showMsg("Rendering...", "info");
      pollPreview();
    });
};

document.getElementById('setDither').onclick = () => {
  const alg = document.getElementById('ditherSelect').value;
  showMsg("Applying dithering...", "info");
  fetch('/mode/dither/set', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({algorithm: alg})
  }).then(()=>{
    showMsg("Rendering...", "info");
    pollPreview();
  });
};

document.getElementById('clearBtn').onclick = () => {
  showMsg("Clearing screen...", "info");
  fetch('/clear', {method:'POST'})
    .then(()=> showMsg("Screen cleared", "success"));
};

document.getElementById('setArt').onclick = () => {
  showMsg("Art mode...", "info");
  fetch('/mode/art/set', {method:'POST'}).then(()=>{
    showMsg("Done (no image).", "warning");
  });
};

document.getElementById('rotateBtn').onclick = () => {
  showMsg("Rotating...", "info");
  fetch('/rotate', {method:'GET'}).then(()=>{
    showMsg("Rendering...", "info");
    pollPreview();
  });
};

// Initialize
loadPool();
//pollPreview();
loadConfig();
</script>
</body>
</html>
"""

# ==== Flask Endpoints ====

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/preview', methods=['GET'])
def preview():
    return jsonify(rendered_image=rendered_image_data,
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
    config['mode']         = 'single'
    config['single_image'] = fn
    save_config()
    threading.Thread(target=lambda: update_epaper_thread(process_image(path)), daemon=True).start()
    return jsonify(success=True)

@app.route('/mode/pool/add', methods=['POST'])
def pool_add():
    files = request.files.getlist('images')
    for f in files:
        fn = secure_filename(f.filename)
        path = os.path.join(pool_dir, fn)
        f.save(path)
        if fn not in config['pool_images']:
            config['pool_images'].append(fn)
    save_config()
    return jsonify(success=True)

@app.route('/pool/list', methods=['GET'])
def pool_list():
    return jsonify(config.get('pool_images', []))

@app.route('/mode/pool/remove', methods=['POST'])
def pool_remove():
    data = request.get_json() or {}
    fn = data.get('filename')
    if fn in config['pool_images']:
        config['pool_images'].remove(fn)
        save_config()
        p = os.path.join(pool_dir, fn)
        if os.path.exists(p):
            os.remove(p)
    return jsonify(success=True)

@app.route('/mode/pool/set', methods=['POST'])
def set_pool_mode():
    if not config['pool_images']:
        return jsonify(error="No images in pool"), 400
    config['mode'] = 'pool'
    save_config()
    fn = random.choice(config['pool_images'])
    path = os.path.join(pool_dir, fn)
    threading.Thread(target=lambda: update_epaper_thread(process_image(path)), daemon=True).start()
    return jsonify(success=True)

@app.route('/mode/art/set', methods=['POST'])
def set_art_mode():
    config['mode'] = 'art'
    save_config()
    return jsonify(success=True)

@app.route('/mode/fit/set', methods=['POST'])
def set_fit_mode():
    data = request.get_json() or {}
    mode = data.get("fit_mode", "pad")
    if mode not in ("pad", "zoom", "stretch"):
        return jsonify(error="Invalid fit mode"), 400

    config['fit_mode'] = mode
    save_config()

    # Reprocess and re-render current image from disk
    try:
        if config['mode'] == 'single' and config['single_image']:
            path = os.path.join(single_dir, config['single_image'])
        elif config['mode'] == 'pool' and config['pool_images']:
            path = os.path.join(pool_dir, config['pool_images'][0])
        else:
            return jsonify(success=True)  # nothing to render

        image = process_image(path)
        threading.Thread(target=lambda: update_epaper_thread(image), daemon=True).start()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/mode/dither/set', methods=['POST'])
def set_dither():
    data = request.get_json() or {}
    alg  = data.get('algorithm')
    if alg in ("floyd-steinberg","atkinson","shiau-fan-2","jarvis-judice-ninke","stucki","burkes"):
        config['dithering'] = alg
        save_config()
        # re-render current image
        if current_image is not None:
            threading.Thread(target=lambda: update_epaper_thread(current_image), daemon=True).start()
        return jsonify(success=True)
    return jsonify(error="Invalid algorithm"), 400

@app.route('/rotate', methods=['GET'])
def rotate():
    global rendering_complete
    if current_image is None:
        return jsonify(error="No image to rotate"), 400
    try:
        res = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
        rot = current_image.rotate(90, expand=True)
        rot = ImageOps.fit(rot, get_target_size(), method=res)
        rendering_complete = False
        threading.Thread(target=lambda: update_epaper_thread(rot), daemon=True).start()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/config', methods=['GET'])
def get_config():
    # return the live config object
    return jsonify(config)

@app.route('/clear', methods=['POST'])
def clear_endpoint():
    clear_screen()
    return jsonify(success=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)