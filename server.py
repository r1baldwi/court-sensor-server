"""
Court Occupancy Detection Server
Uses ONNX Runtime for lightweight YOLOv8 person detection.
"""

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
import cv2
import io, time, json, os, zipfile
from pathlib import Path

# ---- Authentication tokens (from environment variables) ----
DEVICE_TOKEN = os.environ.get("DEVICE_TOKEN", "")
API_TOKEN    = os.environ.get("API_TOKEN", "")
ADMIN_TOKEN  = os.environ.get("ADMIN_TOKEN", "")

# ---- Configuration ----
STATUS_FILE    = Path("status.json")
MODEL_PATH     = Path("yolov8n.onnx")
PERSON_CLASS   = 0
CONF_THRESHOLD = 0.3
KEEP_LAST_PHOTO = True

# ---- Confirmation logic state ----
# Tracks first "free" reading per court for occupied→free confirmation
# Structure: {court_id: {"first_free_at": unix_timestamp}}
pending_free_confirm = {}
CONFIRM_WINDOW_SECONDS = 45  # two free readings within this window = confirmed

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

session    = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
INPUT_SIZE = 640

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

latest_photos = {}


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
    top  = (INPUT_SIZE - nh) // 2
    left = (INPUT_SIZE - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]
    return img, scale, top, left


def postprocess(output, scale, pad_top, pad_left, orig_shape):
    pred = output[0][0].transpose()
    boxes_xywh   = pred[:, :4]
    class_scores = pred[:, 4:]
    person_scores = class_scores[:, PERSON_CLASS]
    keep = person_scores > CONF_THRESHOLD
    boxes_xywh    = boxes_xywh[keep]
    person_scores = person_scores[keep]

    if len(boxes_xywh) == 0:
        return []

    cx, cy, w, h = boxes_xywh.T
    x1 = cx - w / 2;  y1 = cy - h / 2
    x2 = cx + w / 2;  y2 = cy + h / 2
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


def annotate(img_bgr, detections, court_id="", timestamp=None):
    out = img_bgr.copy()

    if timestamp:
        ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        cv2.putText(out, f"{court_id} | {ts_str}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(out, f"{court_id} | {ts_str}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    for (box, score) in detections:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"person {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


@app.post("/api/court-photo")
async def court_photo(
    request:        Request,
    x_court_id:     str = Header("court-1"),
    x_chip_temp:    str = Header(None),
    x_device_token: str = Header(None),
    x_send_time:    str = Header(None),
):
    # AUTHENTICATION
    if x_device_token != DEVICE_TOKEN or not DEVICE_TOKEN:
        raise HTTPException(401, "unauthorized")

    # --- METHOD 1 TIMING: log receive time ---
    receive_time = time.time()
    send_time    = float(x_send_time) if x_send_time else None

    if send_time:
        network_lag = receive_time - send_time
        print(f"[TIMING] {x_court_id} | send: {send_time:.3f} | "
              f"receive: {receive_time:.3f} | "
              f"network lag: {network_lag:.3f}s")
    else:
        print(f"[TIMING] {x_court_id} | receive: {receive_time:.3f} "
              f"(no send time header)")

    body = await request.body()
    print(f"Received {len(body)} bytes from {x_court_id}")

    if len(body) < 1000:
        return JSONResponse({"error": "image too small"}, status_code=400)

    arr     = np.frombuffer(body, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse({"error": "decode failed"}, status_code=400)

    img_in, scale, top, left = preprocess(img_bgr)

    # --- METHOD 1 TIMING: measure inference ---
    inference_start = time.time()
    output = session.run(None, {input_name: img_in})
    inference_end   = time.time()
    inference_time  = inference_end - inference_start

    detections   = postprocess(output, scale, top, left, img_bgr.shape)
    person_count = len(detections)
    occupied     = person_count > 0

    print(f"[TIMING] Inference: {inference_time:.3f}s | "
          f"occupied={occupied}, persons={person_count}")

    if send_time:
        total_so_far = inference_end - send_time
        print(f"[TIMING] Total camera→inference-done: {total_so_far:.3f}s")

    # Annotate with timestamp burned in
    now           = int(time.time())
    annotated_img = annotate(img_bgr, detections, x_court_id, now)

    if KEEP_LAST_PHOTO:
        ok, jpeg_bytes = cv2.imencode(".jpg", annotated_img)
        if ok:
            latest_photos[x_court_id] = jpeg_bytes.tobytes()

    # Save every photo if enabled
    SAVE_PHOTOS = os.environ.get("SAVE_PHOTOS", "false").lower() == "true"
    if SAVE_PHOTOS:
        photos_dir   = Path("photos") / x_court_id
        photos_dir.mkdir(parents=True, exist_ok=True)
        status_label = "occupied" if occupied else "free"
        # Tag pending confirmation shots in filename for easy review
        in_confirm   = x_court_id in pending_free_confirm
        confirm_tag  = "_PENDING" if (not occupied and in_confirm) else ""
        filename     = f"{now}_{status_label}_{person_count}p{confirm_tag}.jpg"
        ok, jpeg_bytes = cv2.imencode(".jpg", annotated_img)
        if ok:
            (photos_dir / filename).write_bytes(jpeg_bytes.tobytes())

    # ---- CONFIRMATION LOGIC: occupied→free requires 2 free readings ----
    status       = load_status()
    current      = status.get(x_court_id, {})
    was_occupied = current.get("occupied", False)
    chip_temp_val = float(x_chip_temp) if x_chip_temp else None

    if not occupied:
        if was_occupied:
            # First "free" after "occupied" — start confirmation window
            pending_free_confirm[x_court_id] = {"first_free_at": now}
            print(f"  -> PENDING CONFIRM: first free reading for {x_court_id}, "
                  f"holding status as occupied | temp={x_chip_temp}°C")
            # Don't update status yet — return early keeping court as occupied
            return {"ok": True, "occupied": False,
                    "person_count": person_count,
                    "status_updated": False,
                    "message": "awaiting_confirmation"}

        elif x_court_id in pending_free_confirm:
            # Second "free" reading — check confirmation window
            first_free_at    = pending_free_confirm[x_court_id]["first_free_at"]
            time_since_first = now - first_free_at

            if time_since_first <= CONFIRM_WINDOW_SECONDS:
                # Confirmed free — clear pending and fall through to update
                print(f"  -> CONFIRMED FREE for {x_court_id} "
                      f"after {time_since_first}s | temp={x_chip_temp}°C")
                del pending_free_confirm[x_court_id]
                # Falls through to status update below

            else:
                # Confirmation window expired — restart confirmation
                print(f"  -> CONFIRM WINDOW EXPIRED ({time_since_first}s) "
                      f"for {x_court_id} — restarting | temp={x_chip_temp}°C")
                pending_free_confirm[x_court_id] = {"first_free_at": now}
                return {"ok": True, "occupied": False,
                        "person_count": person_count,
                        "status_updated": False,
                        "message": "confirmation_window_expired_restarting"}
        else:
            # Already in free state — normal update, no confirmation needed
            pass

    else:
        # Court is occupied — cancel any pending confirmation immediately
        if x_court_id in pending_free_confirm:
            print(f"  -> OCCUPIED again for {x_court_id} — "
                  f"cancelling free confirmation | temp={x_chip_temp}°C")
            del pending_free_confirm[x_court_id]

    # Normal status update
    # Reached for: occupied, confirmed free, or already-free→still-free
    status[x_court_id] = {
        "occupied":     occupied,
        "person_count": person_count,
        "updated_at":   now,
        "chip_temp":    chip_temp_val,
    }
    save_status(status)

    print(f"  -> STATUS UPDATED: occupied={occupied}, "
          f"persons={person_count}, temp={x_chip_temp}°C")

    return {"ok": True, "occupied": occupied, "person_count": person_count}


@app.get("/")
def dashboard(token: str = ""):
    if token != ADMIN_TOKEN or not ADMIN_TOKEN:
        raise HTTPException(401, "unauthorized — append ?token=YOUR_ADMIN_TOKEN")

    status = load_status()
    rows   = ""
    for court, s in status.items():
        age   = int(time.time()) - s["updated_at"]
        color = "#d4edda" if s["occupied"] else "#f8d7da"
        has_photo = court in latest_photos

        # Show confirmation pending state if applicable
        confirm_note = ""
        if court in pending_free_confirm:
            elapsed = int(time.time()) - pending_free_confirm[court]["first_free_at"]
            confirm_note = (f'<p style="color:orange;font-weight:bold;">'
                            f'⏳ Confirming free status... ({elapsed}s elapsed, '
                            f'max {CONFIRM_WINDOW_SECONDS}s)</p>')

        img_tag = (f'<img src="/latest/{court}?token={token}" '
                   f'style="max-width:600px;">') if has_photo else ""

        rows += f"""
        <div style="background:{color};padding:1em;margin:1em 0;border-radius:8px;">
          <h2>{court}: {"OCCUPIED" if s["occupied"] else "free"}</h2>
          <p>{s["person_count"]} person(s), {age}s ago</p>
          <p>Chip temp: {s.get("chip_temp", "N/A")}°C</p>
          {confirm_note}
          {img_tag}
        </div>
        """

    photos_path = Path("photos")
    saved_count = (sum(1 for _ in photos_path.rglob("*.jpg"))
                   if photos_path.exists() else 0)
    saving_on   = os.environ.get("SAVE_PHOTOS", "false").lower() == "true"
    save_line   = (f"<p><small>Photo saving: "
                   f"{'ON' if saving_on else 'OFF'} — "
                   f"{saved_count} photos saved | "
                   f"<a href='/admin/photos/download.zip?token={token}'>"
                   f"Download zip</a></small></p>")

    html = f"""
    <html>
    <head>
      <meta http-equiv="refresh" content="120">
      <title>Court status</title>
    </head>
    <body style="font-family:sans-serif;max-width:700px;margin:2em auto;">
      <h1>Court occupancy</h1>
      {save_line}
      {rows or "<p>No data yet. Waiting for first upload...</p>"}
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/latest/{court_id}")
def latest_photo(court_id: str, token: str = ""):
    if token != ADMIN_TOKEN or not ADMIN_TOKEN:
        raise HTTPException(401, "unauthorized")
    if court_id not in latest_photos:
        return Response(status_code=404)
    return Response(content=latest_photos[court_id], media_type="image/jpeg")


@app.get("/api/status")
def status(x_api_token: str = Header(None)):
    if x_api_token != API_TOKEN or not API_TOKEN:
        raise HTTPException(401, "unauthorized")
    return load_status()


@app.get("/admin/photos/download.zip")
def download_all_photos(token: str = ""):
    if token != ADMIN_TOKEN or not ADMIN_TOKEN:
        raise HTTPException(401, "unauthorized")

    photos_path = Path("photos")
    if not photos_path.exists():
        raise HTTPException(404, "no photos saved yet")

    zip_buffer = io.BytesIO()
    file_count = 0
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for photo_file in photos_path.rglob("*.jpg"):
            zf.write(photo_file, photo_file.relative_to(photos_path))
            file_count += 1

    if file_count == 0:
        raise HTTPException(404, "no photos found")

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition":
                 f"attachment; filename=photos_{int(time.time())}.zip"}
    )


@app.get("/admin/photos/count")
def count_photos(token: str = ""):
    if token != ADMIN_TOKEN or not ADMIN_TOKEN:
        raise HTTPException(401, "unauthorized")

    photos_path = Path("photos")
    if not photos_path.exists():
        return {"total": 0, "by_court": {},
                "saving_enabled": os.environ.get(
                    "SAVE_PHOTOS", "false").lower() == "true"}

    by_court = {}
    total    = 0
    for court_dir in photos_path.iterdir():
        if court_dir.is_dir():
            count = sum(1 for _ in court_dir.glob("*.jpg"))
            by_court[court_dir.name] = count
            total += count

    return {
        "total":          total,
        "by_court":       by_court,
        "saving_enabled": os.environ.get("SAVE_PHOTOS", "false").lower() == "true"
    }


@app.get("/healthz")
def health():
    return {"ok": True}