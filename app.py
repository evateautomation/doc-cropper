import io, zipfile, time
from typing import List, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2, numpy as np

app = FastAPI(title="Doc Cropper API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order quad points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _warp(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = _order_quad(quad.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def detect_documents(bgr: np.ndarray, min_area: int = 20000) -> List[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quads.append(approx)
    # Largest first to avoid tiny noise
    quads.sort(key=lambda q: cv2.contourArea(q), reverse=True)
    return quads

@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}

@app.post("/process")
async def process(file: UploadFile = File(...), return_zip: bool = True):
    # safety limits
    if file.size and file.size > 15 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 15MB).")

    content = await file.read()
    image_array = np.frombuffer(content, dtype=np.uint8)
    bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Invalid image file.")

    quads = detect_documents(bgr)

    crops = []
    for i, q in enumerate(quads, start=1):
        warp = _warp(bgr, q)
        ok, buf = cv2.imencode(".jpg", warp, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            continue
        crops.append((f"document_{i}.jpg", buf.tobytes()))

    if not crops:
        return JSONResponse({"documents_found": 0, "files": []})

    if return_zip:
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in crops:
                zf.writestr(name, data)
        mem.seek(0)
        return StreamingResponse(mem, media_type="application/zip",
                                 headers={"Content-Disposition": 'attachment; filename="documents.zip"'})

    # else return JSON with base64 inline (not recommended for large files)
    import base64
    files64 = [{"name": n, "b64": base64.b64encode(d).decode()} for n, d in crops]
    return {"documents_found": len(crops), "files": files64}
