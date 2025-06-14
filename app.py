#!/usr/bin/env python3

import sys
import os
import time
import threading
import io
import base64
import json
import random

from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance

# allow local driver module in lib/
libdir = os.path.join(os.path.dirname(__file__), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)
import epd13in3E  # Waveshare 13.3" “E” driver

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
    "update_count":  0
}

def load_config():
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config.copy()
    with open(config_path, 'r') as f:
        data = json.load(f)
    # fill in missing keys if any
    for k,v in default_config.items():
        if k not in data:
            data[k] = v
    return data

def save_config():
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

config = load_config()

# ==== E-Paper Setup ====
epd = epd13in3E.EPD()
try:
    epd.Init()
    epd.sleep()  # don't clear on startup
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
def apply_dithering(image, algorithm):
    if algorithm == "floyd-steinberg":
        return image.convert("RGB").convert("P", palette=_palette_img, dither=Image.FLOYDSTEINBERG)
    if algorithm == "atkinson":
        return atkinson_dither(image)
    if algorithm == "shiau-fan-2":
        return shiaufan2_dither(image)
    if algorithm == "jarvis-judice-ninke":
        return jarvis_judice_ninke_dither(image)
    if algorithm == "stucki":
        return stucki_dither(image)
    if algorithm == "burkes":
        return burkes_dither(image)
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

def jarvis_judice_ninke_dither(image):
    kernel = [
        [0,0,   7,   5,   3],
        [3,5,   7,   5,   3],
        [1,3,   5,   3,   1]
    ]
    return error_diffusion(image, kernel, divisor=48, anchor=(2,0))

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

# ==== Display Logic with persisted counter ====
def display_image(image):
    # increment & persist update_count
    config['update_count'] += 1
    cnt = config['update_count']
    save_config()

    # full clear on every 6th update
    if cnt % 6 == 0:
        epd.Init()
        epd.Clear()
        epd.sleep()

    # standard update
    epd.Init()
    img = ImageOps.fit(image, get_target_size(), method=Image.Resampling.LANCZOS)
    dithered = apply_dithering(img, config.get("dithering", "floyd-steinberg"))
    buf = epd.getbuffer(dithered)
    epd.display(buf)
    epd.sleep()
    return dithered

def process_image(src):
    img = Image.open(src).convert("RGB")
    img = ImageOps.exif_transpose(img)
    if img.height > img.width:
        img = img.rotate(90, expand=True)
    img = img.rotate(180, expand=True)
    img = img.resize(get_target_size(), Image.Resampling.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Color(img).enhance(3.0)
    return img

current_image       = None
rendered_image_data = None
rendering_complete  = False

def update_epaper_thread(image):
    global current_image, rendered_image_data, rendering_complete
    try:
        d = display_image(image)
        current_image = d
        buf = io.BytesIO()
        d.rotate(180).save(buf, format="PNG")
        rendered_image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
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
        p  = os.path.join(pool_dir, fn)
        if os.path.exists(p):
            update_epaper_thread(process_image(p))

threading.Thread(target=initial_display, daemon=True).start()

# ==== Flask & Inactivity ====
app = Flask(__name__)
#last_activity = time.time()
#TIMEOUT       = 20*60

@app.before_request
#def touch_activity():
#    global last_activity
#    last_activity = time.time()

#def watchdog():
#    while True:
#        time.sleep(60)
#        if time.time() - last_activity > TIMEOUT:
#            os.system("sudo shutdown now")
#            break

#threading.Thread(target=watchdog, daemon=True).start()


# ==== Web UI Template ====

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Spectra 6 Picture Frame</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <style>
    body { padding: 20px; }
    .section { margin-bottom: 30px; }
  </style>
</head>
<body>
  <h1 class="mb-4">Spectra 6 Picture Frame</h1>

  <div class="section">
    <h3>Single Image Mode</h3>
    <form id="singleForm">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit" class="btn btn-primary mt-2">Upload & Set</button>
    </form>
  </div>

  <div class="section">
    <h3>Image Pool Mode</h3>
    <form id="poolForm">
      <input type="file" name="images" accept="image/*" multiple required>
      <button type="submit" class="btn btn-secondary mt-2">Add Images to Pool</button>
    </form>
    <button id="usePoolBtn" class="btn btn-primary mt-2">Use Pool Mode</button>
    <ul id="poolList"></ul>
  </div>

  <div class="section">
    <h3>Dithering Algorithm</h3>
    <select id="ditherSelect" class="form-control" style="max-width: 300px;">
      <option value="floyd-steinberg">Floyd–Steinberg</option>
      <option value="atkinson">Atkinson</option>
      <option value="shiau-fan-2">Shiau-Fan 2</option>
      <option value="jarvis-judice-ninke">Jarvis–Judice–Ninke</option>
      <option value="stucki">Stucki</option>
      <option value="burkes">Burkes</option>
    </select>
    <button id="setDitherBtn" class="btn btn-primary mt-2">Set Dithering</button>
  </div>

  <div class="section">
    <h3>Clear Screen</h3>
    <button id="clearBtn" class="btn btn-danger">Clear Display</button>
  </div>

  <div class="section">
    <h3>Live Preview</h3>
    <div id="msg"></div>
    <img id="preview" src="/preview.png" class="img-fluid" style="max-width: 400px;">
    <button id="rotateBtn" class="btn btn-warning mt-2">Rotate 90°</button>
  </div>

  <div class="section">
    <h3>Current Configuration</h3>
    <pre id="configDisplay" class="bg-light p-3"></pre>
  </div>

<script>
  function showMsg(txt, cls="info"){
    document.getElementById('msg').innerHTML =
      `<div class="alert alert-${cls}">${txt}</div>`;
  }

  function loadConfig(){
    fetch('/config').then(r=>r.json()).then(c=>{
      document.getElementById('configDisplay').innerText =
        JSON.stringify(c, null, 2);
    });
    fetch('/pool/list').then(r=>r.json()).then(p=>{
      let ul = document.getElementById('poolList');
      ul.innerHTML = p.map(i=>`<li>${i}</li>`).join('');
    });
  }

  document.getElementById('singleForm').onsubmit = e => {
    e.preventDefault();
    let fd = new FormData(e.target);
    fetch('/mode/single', {method:'POST', body:fd})
      .then(() => showMsg("Rendering... Please refresh manually.", "info"));
  };

  document.getElementById('poolForm').onsubmit = e => {
    e.preventDefault();
    let fd = new FormData(e.target);
    fetch('/mode/pool/add', {method:'POST', body:fd})
      .then(() => showMsg("Image(s) added. Please refresh manually.", "success"))
      .then(loadConfig);
  };

  document.getElementById('usePoolBtn').onclick = ()=>{
    fetch('/mode/pool/set', {method:'POST'})
      .then(() => showMsg("Rendering... Please refresh manually.", "info"));
  };

  document.getElementById('setDitherBtn').onclick = ()=>{
    let alg = document.getElementById('ditherSelect').value;
    fetch('/mode/dither/set',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({algorithm: alg})
    }).then(() => showMsg("Rendering... Please refresh manually.", "info"));
  };

  document.getElementById('clearBtn').onclick = ()=>{
    fetch('/clear', {method:'POST'}).then(() => {
      showMsg("Display cleared. Please refresh manually.", "warning");
    });
  };

  document.getElementById('rotateBtn').onclick = ()=>{
    fetch('/rotate').then(() => {
      showMsg("Rotating... Please refresh manually.", "info");
    });
  };

  window.onload = loadConfig;
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
    threading.Thread(target=lambda: (epd.Init(), epd.Clear(), epd.sleep()),
                     daemon=True).start()
    return jsonify(success=True)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
