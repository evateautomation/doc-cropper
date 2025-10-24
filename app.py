import io
import os
import time
import uuid
import hmac
import zipfile
from typing import List, Tuple, Literal

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

app = FastAPI(title="Doc Cropper API", version="1.3.0")

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

def _bbox(cnt_or_quad: np.ndarray):
    # works for contour or 4-pt quad
    if cnt_or_quad.ndim == 3 and cnt_or_quad.shape[0] == 4:
        xs = cnt_or_quad[:, 0, 0]
        ys = cnt_or_quad[:, 0, 1]
        x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
        return x, y, w, h
    else:
        x, y, w, h = cv2.boundingRect(cnt_or_quad)
        return x, y, w, h

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

def _find_candidate_quads(img_bgr: np.ndarray) -> List[np.ndarray]:
    h, w = img_bgr.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1) Adaptive threshold + open to separate touching cards
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k5, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k5, iterations=1)
    c1, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2) Canny fallback
    edges = cv2.Canny(blur, 60, 150)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3, iterations=1)
    c2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = c1 + c2

    quads_raw: List[np.ndarray] = []
    for c in contours:
        area = cv2.contourArea(c)
        # reject tiny and very big regions (avoid whole-scene crop)
        if area < 0.015 * img_area or area > 0.45 * img_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        quad = approx if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(c)).astype("int").reshape(-1, 1, 2)

        x, y, ww, hh = _bbox(quad)
        if ww <= 0 or hh <= 0:
            continue

        # 3) "card-like" filters
        aspect = max(ww / hh, hh / ww)
        if not (1.3 <= aspect <= 2.1):   # ID cards ~ 1.4–1.7; allow a bit wider range
            continue

        rectangularity = area / float(ww * hh)  # 1.0 is a perfect rectangle
        if rectangularity < 0.65:
            continue

        quads_raw.append((quad, (x, y, ww, hh), area))

    # sort tightest/most confident first (smaller bbox & higher rectangularity)
    quads_raw.sort(key=lambda t: (t[2], - (t[2] / (t[1][2] * t[1][3]))))  # ascending by area, prefer tighter

    # 4) Non-maximum suppression by IoU: drop large boxes that cover smaller ones
    selected: List[np.ndarray] = []
    bboxes: List[Tuple[int, int, int, int]] = []
    for quad, box, _ in quads_raw:
        keep = True
        for sb, _ in zip(bboxes, selected):
            if _iou(box, sb) > 0.6:
                keep = False
                break
        if keep:
            selected.append(quad)
            bboxes.append(box)

    # If we still detected more than 4, keep the top 4 by area (descending)
    selected.sort(key=lambda q: cv2.contourArea(q), reverse=True)
    return selected[:4]

def process_image_to_crops(bgr: np.ndarray) -> List[Tuple[str, bytes]]:
    quads = _find_candidate_quads(bgr)
    out: List[Tuple[str, bytes]] = []
    for i, q in enumerate(quads, start=1):
        warped = _warp_perspective(bgr, q)
        upright = auto_upright(warped)
        ok, buf = cv2.imencode(".jpg", upright, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            out.append((f"document_{i}.jpg", buf.tobytes()))
    return out


def _multipart_mixed(files: List[Tuple[str, bytes]]) -> Response:
    boundary = f"BOUNDARY-{uuid.uuid4().hex}"
    bio = io.BytesIO()
    for fname, data in files:
        bio.write(f"--{boundary}\r\n".encode())
        bio.write(
            (
                "Content-Type: image/jpeg\r\n"
                f'Content-Disposition: attachment; filename="{fname}"\r\n'
                "Content-Transfer-Encoding: binary\r\n\r\n"
            ).encode()
        )
        bio.write(data)
        bio.write(b"\r\n")
    bio.write(f"--{boundary}--\r\n".encode())
    bio.seek(0)
    return Response(content=bio.read(), media_type=f"multipart/mixed; boundary={boundary}")

def _zip_response(files: List[Tuple[str, bytes]], zip_name: str) -> StreamingResponse:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files:
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
    _auth: None = Depends(bearer_auth),  # protect this route
):
    if file.size and file.size > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20MB).")

    buf = await file.read()
    arr = np.frombuffer(buf, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Invalid image file.")

    files = process_image_to_crops(bgr)
    if not files:
        return JSONResponse({"documents_found": 0, "files": []})

    if output == "multipart":
        return _multipart_mixed(files)
    if output == "zip":
        return _zip_response(files, zip_name)

    # default: JSON (base64)
    import base64
    payload = [{"name": n, "b64": base64.b64encode(d).decode("utf-8")} for n, d in files]
    return {"documents_found": len(files), "files": payload}

