import io
import os
import time
import uuid
import hmac
import zipfile
from typing import List, Tuple, Literal, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Query,
    Depends, Security
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
API_TOKEN = os.getenv("API_TOKEN", "")  # set in Coolify env vars

app = FastAPI(title="Doc Cropper API", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Swagger security scheme → shows the "Authorize" button
bearer_scheme = HTTPBearer(auto_error=False)

# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────
def bearer_auth(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="Server not configured with API_TOKEN")
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = credentials.credentials.strip()
    if not hmac.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")


# ──────────────────────────────────────────────────────────────────────────────
# OpenCV helpers
# ──────────────────────────────────────────────────────────────────────────────
def _order_quad(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _warp_perspective(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = _order_quad(quad.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = max(int(max(widthA, widthB)), 200)
    maxHeight = max(int(max(heightA, heightB)), 200)
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)

def _score_upright(img_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    edges = cv2.Canny(g, 80, 160)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    row_var = float(np.var(edges.sum(axis=1)) / (edges.shape[0] + 1e-6))
    col_var = float(np.var(edges.sum(axis=0)) / (edges.shape[1] + 1e-6))
    return row_var - 0.6 * col_var

def auto_upright(img_bgr: np.ndarray) -> np.ndarray:
    candidates = [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    scores = [_score_upright(c) for c in candidates]
    return candidates[int(np.argmax(scores))]

def _bbox_from_quad(quad: np.ndarray):
    xs = quad[:, 0, 0]
    ys = quad[:, 0, 1]
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return x1, y1, x2 - x1, y2 - y1

def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)

def _avg_hash(img_bgr: np.ndarray, size: int = 16) -> str:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    mean = g.mean()
    return "".join("1" if p > mean else "0" for p in g.flatten())

# ——— detection on a single image ———
def _find_candidate_quads_single(img_bgr: np.ndarray) -> List[np.ndarray]:
    h, w = img_bgr.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    quads: List[np.ndarray] = []

    # Path A: adaptive threshold
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 9)
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k7, 1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k7, 1)
    c1, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Path B: edges
    edges = cv2.Canny(blur, 60, 150)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3, 1)
    c2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in (c1 + c2):
        area = cv2.contourArea(c)
        if area < 0.008 * img_area or area > 0.75 * img_area:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype("int").reshape(-1, 1, 2)
        x, y, ww, hh = _bbox_from_quad(box)
        if ww <= 0 or hh <= 0:
            continue

        # Looser but card-ish gating
        aspect = max(ww / hh, hh / ww)
        if not (1.15 <= aspect <= 2.6):
            continue
        rectangularity = area / float(ww * hh)
        if rectangularity < 0.5:
            continue

        quads.append(box)

    # NMS to drop overlaps
    if not quads:
        return []
    bboxes = [ _bbox_from_quad(q) for q in quads ]
    keep = []
    for i, q in enumerate(quads):
        bi = bboxes[i]
        ok = True
        for j in range(len(keep)):
            if _iou(bi, bboxes[keep[j]]) > 0.55:
                ok = False
                break
        if ok:
            keep.append(i)

    selected = [quads[i] for i in keep]
    selected.sort(key=lambda q: cv2.contourArea(q), reverse=True)
    return selected[:6]  # cap

# ——— try multiple global rotations; return crops only ———
def detect_and_crop_all_rotations(bgr: np.ndarray, max_cards: int = 6,
                                  want_debug: bool = False) -> Tuple[List[Tuple[str, bytes]], Optional[bytes]]:
    rotations = [
        (bgr, "r0"),
        (cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), "r90"),
        (cv2.rotate(bgr, cv2.ROTATE_180), "r180"),
        (cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), "r270"),
    ]
    out: List[Tuple[str, bytes]] = []
    seen_hashes = set()

    debug_overlay = None
    if want_debug:
        # start with a blank overlay based on original image
        overlay = bgr.copy()

    for img_rot, tag in rotations:
        quads = _find_candidate_quads_single(img_rot)
        for q_idx, q in enumerate(quads, start=1):
            warped = _warp_perspective(img_rot, q)
            upright = auto_upright(warped)
            h = _avg_hash(upright)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            ok, buf = cv2.imencode(".jpg", upright, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                out.append((f"document_{len(out)+1}.jpg", buf.tobytes()))

        if len(out) >= max_cards:
            break

        if want_debug and len(quads) > 0 and tag == "r0":
            # Draw boxes only for the original orientation for simplicity
            dbg = bgr.copy()
            for q in quads:
                pts = q.reshape(-1, 2)
                cv2.polylines(dbg, [pts], True, (0, 255, 0), 3)
            ok, buf = cv2.imencode(".jpg", dbg, [cv2.IMWRITE_JPEG_QUALITY, 85])
            debug_overlay = buf.tobytes() if ok else None

    return out, debug_overlay


# ──────────────────────────────────────────────────────────────────────────────
# Responses
# ──────────────────────────────────────────────────────────────────────────────
def _multipart_mixed(files: List[Tuple[str, bytes]], extra: List[Tuple[str, bytes]] = None) -> Response:
    boundary = f"BOUNDARY-{uuid.uuid4().hex}"
    bio = io.BytesIO()
    def write_part(name, data, content_type="image/jpeg"):
        bio.write(f"--{boundary}\r\n".encode())
        bio.write(
            (
                f"Content-Type: {content_type}\r\n"
                f'Content-Disposition: attachment; filename="{name}"\n'
                "Content-Transfer-Encoding: binary\r\n\r\n"
            ).encode()
        )
        bio.write(data)
        bio.write(b"\r\n")
    for fname, data in files:
        write_part(fname, data)
    if extra:
        for fname, data in extra:
            write_part(fname, data, content_type="image/jpeg")
    bio.write(f"--{boundary}--\r\n".encode())
    bio.seek(0)
    return Response(content=bio.read(), media_type=f"multipart/mixed; boundary={boundary}")

def _zip_response(files: List[Tuple[str, bytes]], zip_name: str, extra: List[Tuple[str, bytes]] = None) -> StreamingResponse:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files:
            zf.writestr(fname, data)
        if extra:
            for fname, data in extra:
                zf.writestr(fname, data)
    mem.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{zip_name}"'}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    output: Literal["json", "multipart", "zip"] = Query("json", description="Response format"),
    zip_name: str = Query("documents.zip", description="Name of the zip (when output=zip)"),
    debug: bool = Query(False, description="Include a debug overlay image of detections"),
    _auth: None = Depends(bearer_auth),  # protect this route
):
    if file.size and file.size > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20MB).")

    buf = await file.read()
    arr = np.frombuffer(buf, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Invalid image file.")

    files, dbg = detect_and_crop_all_rotations(bgr, want_debug=debug)

    if not files:
        return JSONResponse({"documents_found": 0, "files": []})

    debug_parts = []
    if debug and dbg is not None:
        debug_parts = [("debug_overlay.jpg", dbg)]

    if output == "multipart":
        return _multipart_mixed(files, extra=debug_parts)
    if output == "zip":
        return _zip_response(files, zip_name, extra=debug_parts)

    # default: JSON (base64)
    import base64
    payload = [{"name": n, "b64": base64.b64encode(d).decode("utf-8")} for n, d in files]
    if debug and dbg is not None:
        payload.append({"name": "debug_overlay.jpg", "b64": base64.b64encode(dbg).decode("utf-8")})
    return {"documents_found": len(files), "files": payload}
