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
MODEL_PATH     = Path("yolov8s.onnx")
PERSON_CLASS   = 0
CONF_THRESHOLD = 0.15
KEEP_LAST_PHOTO = True

# ---- Confirmation logic state ----
pending_free_confirm = {}
CONFIRM_WINDOW_SECONDS = 180

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


def apply_clahe(img_bgr):
    """Apply CLAHE contrast enhancement to boost visibility in dark/shadowed areas."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(img_bgr):
    # Keep rows 15% to 70% — removes sky on top AND foreground on bottom
    h_full = img_bgr.shape[0]
    crop_top_px = int(h_full * 0.15)
    img_bgr = img_bgr[crop_top_px:int(h_full * 0.70), :]

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
    return img, scale, top, left, crop_top_px


def postprocess(output, scale, pad_top, pad_left, orig_shape, crop_top_px=0):
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

    # Remove letterbox padding offset
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale

    # Add back the crop offset so coordinates are in original image space
    boxes[:, [1, 3]] += crop_top_px

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


def run_inference(img_bgr):
    """Run YOLO inference on an image. Returns (detections, person_count, occupied)."""
    img_in, scale, top, left, crop_top_px = preprocess(img_bgr)
    output     = session.run(None, {input_name: img_in})
    detections = postprocess(output, scale, top, left, img_bgr.shape, crop_top_px)
    person_count = len(detections)
    return detections, person_count, person_count > 0


def annotate(img_bgr, detections, court_id="", timestamp=None, clahe_used=False):
    out = img_bgr.copy()

    if timestamp:
        ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        label  = f"{court_id} | {ts_str}"
        if clahe_used:
            label += " [CLAHE]"
        cv2.putText(out, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(out, label, (10, 25),
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

    # --- TIMING: log receive time ---
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

    # ---- PASS 1: standard inference ----
    inference_start = time.time()
    detections, person_count, occupied = run_inference(img_bgr)
    inference_end = time.time()
    print(f"[INFERENCE] Pass 1 | {inference_end - inference_start:.3f}s | "
          f"occupied={occupied}, persons={person_count}")

    clahe_used = False

    # ---- PASS 2: CLAHE enhancement if no person detected ----
    if not occupied:
        print(f"[CLAHE] No person detected on pass 1 — applying CLAHE for pass 2")
        clahe_start = time.time()
        img_clahe   = apply_clahe(img_bgr)
        detections_clahe, person_count_clahe, occupied_clahe = run_inference(img_clahe)
        clahe_end   = time.time()
        print(f"[CLAHE] Pass 2 | {clahe_end - clahe_start:.3f}s | "
              f"occupied={occupied_clahe}, persons={person_count_clahe}")

        if occupied_clahe:
            print(f"[CLAHE] ✓ CLAHE rescued detection for {x_court_id} — "
                  f"{person_count_clahe} person(s) found on pass 2")
            detections  = detections_clahe
            person_count = person_count_clahe
            occupied    = True
            clahe_used  = True
        else:
            print(f"[CLAHE] No person detected on pass 2 either — reporting free")

    if send_time:
        total_so_far = time.time() - send_time
        print(f"[TIMING] Total camera→inference-done: {total_so_far:.3f}s")

    # Annotate original image (note [CLAHE] in label if pass 2 detected)
    now           = int(time.time())
    annotated_img = annotate(img_bgr, detections, x_court_id, now, clahe_used)

    if KEEP_LAST_PHOTO:
        ok, jpeg_bytes = cv2.imencode(".jpg", annotated_img)
        if ok:
            latest_photos[x_court_id] = jpeg_bytes.tobytes()

    # Save photos if enabled
    SAVE_PHOTOS = os.environ.get("SAVE_PHOTOS", "false").lower() == "true"
    if SAVE_PHOTOS:
        status_label = "occupied" if occupied else "free"
        in_confirm   = x_court_id in pending_free_confirm
        confirm_tag  = "_PENDING" if (not occupied and in_confirm) else ""
        clahe_tag    = "_CLAHE" if clahe_used else ""
        filename     = f"{now}_{status_label}_{person_count}p{confirm_tag}{clahe_tag}.jpg"

        # Save raw (clean input for replay testing)
        raw_dir = Path("photos") / x_court_id / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        ok_raw, raw_bytes = cv2.imencode(".jpg", img_bgr)
        if ok_raw:
            (raw_dir / filename).write_bytes(raw_bytes.tobytes())

        # Save annotated (for admin dashboard review)
        ann_dir = Path("photos") / x_court_id / "annotated"
        ann_dir.mkdir(parents=True, exist_ok=True)
        ok_ann, ann_bytes = cv2.imencode(".jpg", annotated_img)
        if ok_ann:
            (ann_dir / filename).write_bytes(ann_bytes.tobytes())

    # ---- CONFIRMATION LOGIC: occupied→free requires 2 free readings ----
    status       = load_status()
    current      = status.get(x_court_id, {})
    was_occupied = current.get("occupied", False)
    chip_temp_val = float(x_chip_temp) if x_chip_temp else None

    if not occupied:
        if x_court_id in pending_free_confirm:
            first_free_at    = pending_free_confirm[x_court_id]["first_free_at"]
            time_since_first = now - first_free_at

            if time_since_first <= CONFIRM_WINDOW_SECONDS:
                print(f"  -> CONFIRMED FREE for {x_court_id} "
                      f"after {time_since_first}s | temp={x_chip_temp}°C")
                del pending_free_confirm[x_court_id]
            else:
                print(f"  -> CONFIRM WINDOW EXPIRED ({time_since_first}s) "
                      f"for {x_court_id} — restarting | temp={x_chip_temp}°C")
                pending_free_confirm[x_court_id] = {"first_free_at": now}
                return {"ok": True, "occupied": False,
                        "person_count": person_count,
                        "status_updated": False,
                        "message": "confirmation_window_expired_restarting"}

        elif was_occupied:
            pending_free_confirm[x_court_id] = {"first_free_at": now}
            print(f"  -> PENDING CONFIRM: first free reading for {x_court_id}, "
                  f"holding status as occupied | temp={x_chip_temp}°C")
            return {"ok": True, "occupied": False,
                    "person_count": person_count,
                    "status_updated": False,
                    "message": "awaiting_confirmation"}
        else:
            pass

    else:
        if x_court_id in pending_free_confirm:
            print(f"  -> OCCUPIED again for {x_court_id} — "
                  f"cancelling free confirmation | temp={x_chip_temp}°C")
            del pending_free_confirm[x_court_id]

    # Normal status update
    was_occupied_before = current.get("occupied", False)

    # Only set occupied_since on transition to occupied, preserve it while staying occupied
    if occupied and not was_occupied_before:
        occupied_since = now
    elif occupied and was_occupied_before:
        occupied_since = current.get("occupied_since", now)
    else:
        occupied_since = None

    status[x_court_id] = {
        "occupied":        occupied,
        "person_count":    person_count,
        "updated_at":      now,
        "chip_temp":       chip_temp_val,
        "occupied_since":  occupied_since,
    }
    save_status(status)

    print(f"  -> STATUS UPDATED: occupied={occupied}, "
          f"persons={person_count}, temp={x_chip_temp}°C"
          f"{' [CLAHE]' if clahe_used else ''}")

    return {"ok": True, "occupied": occupied, "person_count": person_count,
            "occupied_since": occupied_since, "clahe_used": clahe_used}


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
            count = sum(1 for _ in court_dir.rglob("*.jpg"))
            by_court[court_dir.name] = count
            total += count

    return {
        "total":          total,
        "by_court":       by_court,
        "saving_enabled": os.environ.get("SAVE_PHOTOS", "false").lower() == "true"
    }


@app.get("/admin/photos/clear")
def clear_photos(token: str = ""):
    if token != ADMIN_TOKEN or not ADMIN_TOKEN:
        raise HTTPException(401, "unauthorized")

    photos_path = Path("photos")
    if not photos_path.exists():
        return {"ok": True, "deleted": 0, "message": "no photos directory found"}

    deleted = 0
    for photo_file in photos_path.rglob("*.jpg"):
        photo_file.unlink()
        deleted += 1

    for court_dir in photos_path.iterdir():
        if court_dir.is_dir() and not any(court_dir.iterdir()):
            court_dir.rmdir()

    print(f"[ADMIN] Cleared {deleted} photos from disk")
    return {"ok": True, "deleted": deleted,
            "message": f"Deleted {deleted} photos. Directory structure preserved."}


@app.get("/healthz")
def health():
    return {"ok": True}