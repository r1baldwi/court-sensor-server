"""
Court Occupancy Detection Server
Uses ONNX Runtime for lightweight YOLOv8 person detection.
Designed to run on Render's free tier.
"""

from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
import onnxruntime as ort
import numpy as np
import cv2
import io, time, json, os
from pathlib import Path

# ---- Configuration ----
STATUS_FILE = Path("status.json")
MODEL_PATH = Path("yolov8n.onnx")
PERSON_CLASS = 0
CONF_THRESHOLD = 0.4
KEEP_LAST_PHOTO = True   # keep most recent photo per court in memory only

# ---- Model setup ----
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Generate it locally with: "
        "python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')\""
    )

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
INPUT_SIZE = 640

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow any origin to read your API
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# In-memory cache of the most recent annotated photo per court
# This avoids writing to disk on Render (ephemeral filesystem)
latest_photos = {}  # court_id -> bytes (annotated JPEG)


def load_status():
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {}


def save_status(s):
    STATUS_FILE.write_text(json.dumps(s, indent=2))


def preprocess(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    top = (INPUT_SIZE - nh) // 2
    left = (INPUT_SIZE - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]  # NCHW
    return img, scale, top, left


def postprocess(output, scale, pad_top, pad_left, orig_shape):
    pred = output[0][0].transpose()  # (8400, 84)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    person_scores = class_scores[:, PERSON_CLASS]
    keep = person_scores > CONF_THRESHOLD
    boxes_xywh = boxes_xywh[keep]
    person_scores = person_scores[keep]

    if len(boxes_xywh) == 0:
        return []

    cx, cy, w, h = boxes_xywh.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale

    H, W = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)

    keep_idx = cv2.dnn.NMSBoxes(
        boxes.tolist(), person_scores.tolist(), CONF_THRESHOLD, 0.45
    )
    if len(keep_idx) == 0:
        return []
    keep_idx = np.array(keep_idx).flatten()
    return [(boxes[i], float(person_scores[i])) for i in keep_idx]


def annotate(img_bgr, detections):
    out = img_bgr.copy()
    for (box, score) in detections:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"person {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


@app.post("/api/court-photo")
async def court_photo(request: Request, x_court_id: str = Header("court-1")):
    body = await request.body()
    print(f"Received {len(body)} bytes from {x_court_id}")

    if len(body) < 1000:
        return JSONResponse({"error": "image too small"}, status_code=400)

    arr = np.frombuffer(body, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse({"error": "decode failed"}, status_code=400)

    img_in, scale, top, left = preprocess(img_bgr)
    output = session.run(None, {input_name: img_in})
    detections = postprocess(output, scale, top, left, img_bgr.shape)

    person_count = len(detections)
    occupied = person_count > 0

    # Annotate and keep in memory (not disk)
    if KEEP_LAST_PHOTO:
        annotated = annotate(img_bgr, detections)
        ok, jpeg_bytes = cv2.imencode(".jpg", annotated)
        if ok:
            latest_photos[x_court_id] = jpeg_bytes.tobytes()

    status = load_status()
    status[x_court_id] = {
        "occupied": occupied,
        "person_count": person_count,
        "updated_at": int(time.time()),
    }
    save_status(status)

    print(f"  -> occupied={occupied}, persons={person_count}")
    return {"ok": True, "occupied": occupied, "person_count": person_count}


@app.get("/")
def dashboard():
    status = load_status()
    rows = ""
    for court, s in status.items():
        age = int(time.time()) - s["updated_at"]
        color = "#d4edda" if s["occupied"] else "#f8d7da"
        has_photo = court in latest_photos
        img_tag = f'<img src="/latest/{court}" style="max-width:600px;">' if has_photo else ""
        rows += f"""
        <div style="background:{color};padding:1em;margin:1em 0;border-radius:8px;">
          <h2>{court}: {"OCCUPIED" if s["occupied"] else "free"}</h2>
          <p>{s["person_count"]} person(s), {age}s ago</p>
          {img_tag}
        </div>
        """
    html = f"""
    <html><head><meta http-equiv="refresh" content="120"><title>Court status</title></head>
    <body style="font-family:sans-serif;max-width:700px;margin:2em auto;">
      <h1>Court occupancy</h1>
      {rows or "<p>No data yet. Waiting for first upload...</p>"}
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/latest/{court_id}")
def latest_photo(court_id: str):
    if court_id not in latest_photos:
        return Response(status_code=404)
    return Response(content=latest_photos[court_id], media_type="image/jpeg")


@app.get("/api/status")
def status():
    return load_status()


@app.get("/healthz")
def health():
    return {"ok": True}