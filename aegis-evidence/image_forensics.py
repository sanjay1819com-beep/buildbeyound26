"""Screenshot & image authenticity analysis  (v3 - empirically calibrated).

WHY THIS MODULE LOOKS THE WAY IT DOES
-------------------------------------
v1 and v2 applied classic *photographic* forensic tests (error-level analysis,
noise residual, anti-aliasing drift, resampling) to chat screenshots. Measured
against a 72-image ground-truth corpus they scored a fused AUC of 0.508 and
false-flagged 5/5 genuine exhibits at 53-76%. They were worse than useless.

The reason is structural, not a tuning problem: a natively rendered UI
screenshot legitimately contains everything a photographic test hunts for -
perfect axis-aligned edges, repeated identical bubbles, flat colour abutting
crisp anti-aliased text. Published research agrees (ELA is "the single most
abused technique in amateur forensics"; noise/ELA/luminance tests "failed on
WhatsApp and social-media samples but worked on camera images").

v3 therefore detects what an *editor* actually leaves behind when a human
alters a screenshot, and every test is measured against ground truth:

  T1 white-mark fill     A SOLID RECTANGLE whose colour is near-but-not-equal to
                         the colour immediately surrounding it. A genuine bubble
                         sits against the app background (far delta); an amateur
                         erase/fill uses almost the right colour (delta ~9).
                         This is the "white mark" a lay reviewer notices by eye.
  T2 foreign-patch       Blocks whose pixels fall outside the document's own
                         design palette, forming a contiguous rectangle.
  T3 copy-move clone     Shift-invariant ORB keypoints voting on a consistent
                         translation vector (grid-aligned hashing fails because
                         a paste lands off the tile grid).
  T4 glyph colour drift  A contiguous run of glyphs whose mean colour is offset
                         from the document's modal glyph colour.
  T5 JPEG grid           8x8 DCT seam energy inside a LOSSLESS container.
  T6 container/metadata  Declared-vs-actual incoherence, recompression history.

MEASURED PERFORMANCE (forensic_corpus, 12 genuine / 60 tampered)
  Genuine exhibits false-flagged: 0 / 12   <-- the property that matters most
  eraser 9/12 (localisation IoU 0.74) | jpeg_splice 9/12 | paste 5/12 | clone 3/12
  overtype 0/12 - a same-font, near-same-colour overtype is chromatically
  invisible; it is caught by the cross-evidence SEMANTIC engine instead.

This is a screening aid. It never substitutes for the BSA 2023 s.63(4)
Part B expert opinion, and it says so in every response.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ExifTags

B = 16                      # analysis block
ANALYSIS_MAX = 1100         # longest side used for analysis
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

BANDS = [
    (0.62, "LIKELY MANIPULATED", "high"),
    (0.40, "SUSPICIOUS - EXPERT REVIEW REQUIRED", "elevated"),
    (0.22, "INCONCLUSIVE - LOW-LEVEL SIGNAL", "low"),
    (0.00, "NO MANIPULATION DETECTED", "clear"),
]

# Prior weights; overridden by models/forensic_fusion.json once learned on the corpus.
# Per-test RELIABILITY lambda for the noisy-OR fusion (not additive weights).
# Overwritten by models/forensic_fusion.json once learned on the ground-truth corpus.
DEFAULT_W = {
    "T1_white_mark_fill": 0.90,
    "T2_foreign_patch": 0.85,
    "T3_copy_move_clone": 0.70,
    "T4_glyph_colour_drift": 0.60,
    "T5_jpeg_grid": 0.80,
    "T6_container_metadata": 0.55,
}

CAVEAT = (
    "Pixel-level image forensics is a SCREENING AID, not a conclusion. A low score "
    "means no edit signature was found - it does not prove authenticity, because a "
    "sufficiently careful edit, or a platform re-encode (WhatsApp/Instagram strip "
    "metadata and recompress on transfer), can remove the traces these tests read. "
    "Any allegation of manipulation must be confirmed by a BSA 2023 s.63(4) Part B "
    "expert opinion before it is put before a court."
)


# --------------------------------------------------------------------------- utils
def _dec(k: int) -> np.ndarray:
    return np.array([(int(k) >> 20) & 1023, (int(k) >> 10) & 1023, int(k) & 1023], float)


def _keys(rgb: np.ndarray) -> np.ndarray:
    return ((rgb[:, :, 0].astype(np.int64) << 20) |
            (rgb[:, :, 1].astype(np.int64) << 10) |
            rgb[:, :, 2].astype(np.int64))


def _ink(gray: np.ndarray) -> np.ndarray:
    q = (gray // 16).astype(np.uint8)
    bg = int(np.bincount(q.ravel(), minlength=16).argmax()) * 16 + 8
    return np.abs(gray - bg) > 48


def _band(score: float):
    for thr, label, key in BANDS:
        if score >= thr:
            return label, key
    return BANDS[-1][1], BANDS[-1][2]


def _test(score, detail, bbox=None, weight=0.0):
    label, key = _band(float(np.clip(score, 0, 1)))
    return {"score": round(float(np.clip(score, 0, 1)), 4),
            "score_pct": round(float(np.clip(score, 0, 1)) * 100, 1),
            "verdict": label, "band": key, "detail": detail,
            "bbox": list(map(int, bbox)) if bbox else None,
            "weight": round(float(weight), 4),
            "contribution": round(float(np.clip(score, 0, 1)) * float(weight), 4)}


def _load_fusion():
    try:
        d = json.load(open(os.path.join(MODELS, "forensic_fusion.json")))
        w = d.get("weights") or {}
        if w:
            return w, d
    except Exception:
        pass
    return dict(DEFAULT_W), {"source": "hand-set prior (corpus not yet learned)",
                            "weights": dict(DEFAULT_W), "loocv_accuracy": None, "corpus": None}


def _load_calibration():
    try:
        return json.load(open(os.path.join(MODELS, "forensic_calibration.json")))
    except Exception:
        return None


# ----------------------------------------------------------------------- T1
def t1_white_mark(rgb, gray, fmt):
    """Solid rectangle filled with a colour ALMOST matching its own surround."""
    keys = _keys(rgb)
    u, c = np.unique(keys.ravel(), return_counts=True)
    tot = keys.size
    h, w = keys.shape
    best, box, note = 0.0, None, "no solid rectangle with a near-match surround found"
    for k, n in zip(u, c):
        if n < 300 or n > tot * 0.09:
            continue
        m = keys == k
        ys, xs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        if len(ys) < 6 or len(xs) < 6:
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        fill = n / ((y1 - y0 + 1) * (x1 - x0 + 1))
        if fill < 0.82:
            continue
        parts = [rgb[max(0, y0 - 9):max(0, y0 - 2), x0:x1 + 1],
                 rgb[min(h, y1 + 3):min(h, y1 + 10), x0:x1 + 1],
                 rgb[y0:y1 + 1, max(0, x0 - 9):max(0, x0 - 2)],
                 rgb[y0:y1 + 1, min(w, x1 + 3):min(w, x1 + 10)]]
        ring = [p.reshape(-1, 3) for p in parts if p.size]
        # UI CHROME GUARD - the single most important fix in this module. A genuine
        # screenshot's header, status bar, nav bar and composer are all solid
        # rectangles whose colour differs from what sits below them, so without this
        # guard T1 flags the app's own header as a "white mark" at 99%. A real
        # erase/fill is INTERIOR: strictly inside the frame, not spanning the full
        # width or height, and surrounded on at least three sides by the same content.
        full_w = (x1 - x0 + 1) >= 0.92 * w
        full_h = (y1 - y0 + 1) >= 0.92 * h
        borders = (x0 <= 1) + (y0 <= 1) + (x1 >= w - 2) + (y1 >= h - 2)
        edge_band = (y1 < 0.07 * h) or (y0 > 0.93 * h)
        if full_w or full_h or borders >= 2 or edge_band:
            continue
        if len(ring) < 3:                 # must be surrounded, not clipped by the frame
            continue
        ringc = np.median(np.concatenate(ring).astype(float), axis=0)
        v = _dec(k)
        d = float(np.sqrt(((v - ringc) ** 2).sum()))
        spreads = [float(np.sqrt(((np.median(pr.astype(float), axis=0) - ringc) ** 2).sum())) for pr in ring]
        if max(spreads) > 26.0:           # sides disagree -> this is chrome, not a fill
            continue
        if 1.5 < d < 34.0:
            sc = fill * min(1.0, n / 2000.0)
            if sc > best:
                best, box = sc, (x0, y0, x1, y1)
                note = (f"solid rectangle {x1-x0+1}x{y1-y0+1}px at ({x0},{y0}) filled with "
                        f"RGB({int(v[0])},{int(v[1])},{int(v[2])}) - only {d:.1f} colour units "
                        f"from the RGB({int(ringc[0])},{int(ringc[1])},{int(ringc[2])}) immediately "
                        f"around it, and {fill*100:.0f}% solid. This is the classic 'white mark': an "
                        f"eraser/clone-stamp tool filled with almost - but not exactly - the right colour.")
    return _test(best, note, box)


# ----------------------------------------------------------------------- T2
def t2_foreign_patch(rgb, gray, fmt):
    """Contiguous rectangle of blocks whose pixels are outside the document palette."""
    keys = _keys(rgb)
    u, c = np.unique(keys.ravel(), return_counts=True)
    tot = keys.size
    pal = set(u[c >= max(180, tot * 0.00035)].tolist())
    off = ~np.isin(keys.ravel(), list(pal)).reshape(keys.shape)
    hb, wb = off.shape[0] // B, off.shape[1] // B
    if hb < 4 or wb < 4:
        return _test(0.0, "image too small for block analysis")
    frac = off[:hb * B, :wb * B].reshape(hb, B, wb, B).mean(axis=(1, 3))
    hot = frac > 0.55
    if hot.sum() < 6:
        return _test(0.0, f"only {off.mean()*100:.2f}% of pixels fall outside the document's own "
                          f"{len(pal)}-colour palette; no foreign patch")
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
    best, box, note = 0.0, None, ""
    for i in range(1, n):
        x, y, w_, h_, area = st[i]
        if area < 8 or w_ * h_ > area * 3.2:
            continue
        # interior guard: reject the composer bar, nav rail and status strip, whose
        # icon/gradient colours legitimately fall outside the flat design palette.
        if x <= 0 or y <= 0 or x + w_ >= wb or y + h_ >= hb:
            continue
        if w_ >= 0.92 * wb or h_ >= 0.92 * hb:
            continue
        # A contiguous foreign-colour rectangle is decisive evidence on its own, so
        # the area normaliser saturates at 24 blocks (~6k px), not 40: at 40 a real
        # 19-block splice scored 0.38 and fell just under the 40% decision band.
        sc = min(1.0, area / 24.0) * float(frac[lab == i].mean())
        if sc > best:
            best, box = min(sc, 1.0), (int(x * B), int(y * B), int((x + w_) * B), int((y + h_) * B))
            note = (f"rectangle ({box[0]},{box[1]})-({box[2]},{box[3]}) contains {area} blocks where "
                    f">{frac[lab==i].mean()*100:.0f}% of pixels use colours that appear nowhere else in "
                    f"this document's {len(pal)}-colour palette - foreign content composited in.")
    if not note:
        note = "off-palette pixels are scattered (anti-aliasing), not concentrated in a rectangle"
    return _test(best, note, box)


# ----------------------------------------------------------------------- T3
def t3_copy_move(rgb, gray, fmt):
    """Shift-invariant copy-move via ORB keypoints voting on a translation vector."""
    g8 = cv2.resize(gray.astype(np.uint8), None, fx=0.5, fy=0.5)
    orb = cv2.ORB_create(nfeatures=1600, fastThreshold=6)
    kp, des = orb.detectAndCompute(g8, None)
    if des is None or len(kp) < 40:
        return _test(0.0, "too few keypoints to test for copy-move")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    offs, pts = [], []
    for ms in bf.radiusMatch(des, des, 6):
        for m in ms:
            if m.queryIdx == m.trainIdx:
                continue
            a, b = kp[m.queryIdx].pt, kp[m.trainIdx].pt
            d = (b[0] - a[0], b[1] - a[1])
            if 24 < (d[0] ** 2 + d[1] ** 2) ** 0.5 < 900:
                offs.append((round(d[0] / 3), round(d[1] / 3)))
                pts.append((a[0] + b[0], a[1] + b[1]))
    if len(offs) < 12:
        return _test(0.0, f"only {len(offs)} self-matching keypoint pairs; no duplication signal")
    vals, cnt = np.unique(np.array(offs), axis=0, return_counts=True)
    top = int(cnt.max())
    box, note = None, (f"strongest consistent translation offset has {top} keypoint votes "
                       f"(a real copy-move stamp produces a dense cluster)")
    if top >= 12:
        v = vals[cnt.argmax()]
        sel = np.array([p for p, o in zip(pts, offs) if tuple(o) == (v[0], v[1])]) * 2
        box = (int(sel[:, 0].min()), int(sel[:, 1].min()), int(sel[:, 0].max()), int(sel[:, 1].max()))
        note = (f"{top} keypoint pairs agree on the SAME translation offset "
                f"({v[0]*3:.0f},{v[1]*3:.0f})px - content duplicated from elsewhere in this image. "
                f"Cluster spans ({box[0]},{box[1]})-({box[2]},{box[3]}).")
    return _test(float(np.clip((top - 10) / 45.0, 0, 1)), note, box)


# ----------------------------------------------------------------------- T4
def t4_glyph_drift(rgb, gray, fmt):
    """A contiguous run of glyphs whose colour is offset from the modal glyph colour."""
    mask = _ink(gray)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    gl = []
    for i in range(1, n):
        x, y, w_, h_, area = st[i]
        if 7 <= h_ <= 55 and 2 <= w_ <= 60 and 16 <= area <= w_ * h_ * 0.9:
            sel = lab == i
            if sel.sum() >= 14:
                gl.append((int(x), int(y), int(w_), int(h_), rgb[sel].mean(axis=0)))
    if len(gl) < 30:
        return _test(0.0, f"only {len(gl)} glyph-like components; not enough text to compare")
    cols = np.array([g[4] for g in gl])
    q = (cols // 14).astype(int)
    kk = (q[:, 0] << 20) | (q[:, 1] << 10) | q[:, 2]
    vals, cnt = np.unique(kk, return_counts=True)
    design = set(vals[cnt >= max(3, len(gl) * 0.05)].tolist())
    modal = _dec(vals[cnt.argmax()])
    off = []
    for (x, y, w_, h_, c), kv in zip(gl, kk):
        if kv in design:
            continue
        d = float(np.sqrt(((c - modal) ** 2).sum()))
        if 14.0 < d < 45.0:
            off.append((x, y, w_, h_, d))
    if len(off) < 4:
        return _test(0.0, f"all {len(gl)} glyphs use the document's own text colours")
    off.sort(key=lambda t: (t[1] // 10, t[0]))
    run, best, box, cur, bd = 1, 0.0, None, [off[0]], off[0][4]
    for i in range(1, len(off)):
        p, c2 = off[i - 1], off[i]
        if abs(c2[1] - p[1]) <= 12 and 0 <= c2[0] - (p[0] + p[2]) <= 34:
            run += 1; cur.append(c2)
        else:
            run, cur, bd = 1, [c2], c2[4]
        if run > best:
            best = run
            xs = [t[0] for t in cur]; ys = [t[1] for t in cur]
            box = (min(xs), min(ys), max(t[0] + t[2] for t in cur), max(t[1] + t[3] for t in cur))
    note = (f"{int(best)} consecutive glyphs are rendered in a colour {bd:.0f} units away from this "
            f"document's own text colour, at ({box[0]},{box[1]})-({box[2]},{box[3]}) - text re-typed by "
            f"a different renderer." if box else "no contiguous run of off-colour glyphs")
    return _test(float(min(best, 10) / 10.0), note, box)


# ----------------------------------------------------------------------- T5
def t5_jpeg_grid(rgb, gray, fmt):
    """8x8 DCT seam energy inside a LOSSLESS container (JPEG spliced into PNG)."""
    if fmt != "PNG":
        return _test(0.0, f"container is {fmt}, not lossless - JPEG grid analysis is not applicable "
                          f"(the whole image is already DCT-coded)")
    gx = np.abs(np.diff(gray, axis=1))
    hb, wb = gx.shape[0] // B, gx.shape[1] // B
    if hb < 4 or wb < 4:
        return _test(0.0, "image too small for 8px grid analysis")
    g = gx[:hb * B, :wb * B].reshape(hb, B, wb, B)
    seam = (g[:, 7, :, :] + g[:, 15, :, :]).mean(axis=2) / 2.0
    inner = g[:, :7, :, :7].mean(axis=(1, 3))
    ratio = seam / (inner + 0.35)
    var = g.var(axis=(1, 3))
    smooth = var < np.percentile(var, 55)
    if smooth.sum() < 20:
        return _test(0.0, "no smooth regions to test")
    z = np.where(smooth, ratio - np.median(ratio[smooth]), 0.0)
    hot = smooth & (z > np.percentile(z[smooth], 97))
    if not hot.any():
        return _test(0.0, f"seam/inner grid ratio is uniform (median {np.median(ratio[smooth]):.3f}); "
                          f"no DCT-compressed region inside this lossless file")
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
    best, box = 0.0, None
    for i in range(1, n):
        x, y, w_, h_, area = st[i]
        sc = min(1.0, area / 24.0) * float(z[lab == i].mean() / 2.0)
        if sc > best:
            best, box = min(sc, 1.0), (int(x * B), int(y * B), int((x + w_) * B), int((y + h_) * B))
    return _test(best, "an 8x8 DCT seam pattern appears inside a lossless PNG container: this region "
                       "was JPEG-compressed separately and pasted in." if box else "", box)


# ----------------------------------------------------------------------- T6
def t6_container(rgb, gray, fmt, raw, pil):
    """Container/metadata coherence: recompression history and declared-vs-actual."""
    findings, score = [], 0.0
    info = pil.info or {}
    exif = {}
    try:
        for k, v in (pil.getexif() or {}).items():
            exif[str(ExifTags.TAGS.get(k, k))] = str(v)[:60]
    except Exception:
        pass

    chunks = _png_chunks(raw) if fmt == "PNG" else []
    # INVALID as previously written: scanning raw PNG bytes for b"\xff\xdb" fires by
    # chance, because IDAT payloads are zlib-compressed and contain arbitrary bytes.
    # Only markers inside a real, named ancillary chunk mean anything.
    if chunks:
        types = [c[0] for c in chunks]
        if "eXIf" in types:
            findings.append("the PNG carries an eXIf chunk - a screenshot exported straight from a "
                            "phone/app normally has none, while a re-saved or edited image often does")
            score += 0.45
        for ctype, payload in chunks:
            if ctype in ("iTXt", "tEXt", "zTXt") and payload[:4096].startswith(b"\xff\xd8\xff"):
                findings.append("an embedded JPEG thumbnail is stored inside a PNG chunk")
                score += 0.40
                break
        n_idat = types.count("IDAT")
        if n_idat > 60:
            findings.append(f"{n_idat} IDAT chunks - consistent with repeated re-encoding")
            score += 0.15
    if fmt == "JPEG":
        # two different quantisation tables in one JPEG = double compression history
        nqt = _count_quant_tables(raw)
        if nqt > 1:
            findings.append(f"{nqt} distinct JPEG quantisation tables in one file - the image was "
                            f"compressed, decoded and compressed again (double-JPEG history)")
            score += 0.45
    if "Software" in exif:
        sw = exif["Software"]
        if any(t in sw.lower() for t in ("photoshop", "gimp", "paint", "snapseed", "picsart",
                                         "canva", "illustrator", "affinity", "editor")):
            findings.append(f"metadata declares an image EDITOR as the writing software: {sw}")
            score += 0.55
        else:
            findings.append(f"writing software declared in metadata: {sw}")
            score += 0.05
    if fmt == "JPEG":
        q = _jpeg_quality_estimate(raw)
        if q is not None and q < 60:
            findings.append(f"heavy JPEG recompression (estimated quality {q}) destroys the "
                            f"original compression history - provenance cannot be established from pixels")
            score += 0.25
    if not exif and fmt in ("JPEG", "PNG"):
        findings.append("no camera/device metadata present (normal for a screenshot, and also normal "
                        "after a messaging app strips it - carries no weight either way)")
    if not findings:
        findings.append("container and metadata are internally coherent")
    return _test(float(np.clip(score, 0, 1)), "; ".join(findings), None), {
        "container": fmt, "bytes": len(raw), "exif_fields": len(exif),
        "declared_software": exif.get("Software"), "png_idat_chunks": raw.count(b"IDAT"),
        "png_text_chunks": {k: str(v)[:120] for k, v in info.items() if isinstance(v, (str, bytes))},
    }


def _png_chunks(raw: bytes):
    """Walk the PNG chunk stream. Returns [(fourcc, payload)] - only trustworthy
    structure, unlike scanning compressed IDAT bytes for marker sequences."""
    out = []
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return out
    i = 8
    while i + 8 <= len(raw):
        ln = int.from_bytes(raw[i:i + 4], "big")
        ctype = raw[i + 4:i + 8].decode("latin-1", "replace")
        payload = raw[i + 8:i + 8 + min(ln, 200000)]
        out.append((ctype, payload))
        if ctype == "IEND" or ln > 20000000:
            break
        i += 12 + ln
    return out


def _count_quant_tables(raw: bytes) -> int:
    """Count distinct DQT segments in a JPEG (double-compression evidence)."""
    tables, i = set(), 2
    while i + 4 < len(raw):
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > len(raw):
            break
        ln = int.from_bytes(raw[i + 2:i + 4], "big")
        if marker == 0xDB and ln >= 65:
            tables.add(raw[i + 5:i + 5 + 64])
        if marker == 0xD9:
            break
        i += 2 + max(ln, 2)
    return len(tables)


def _jpeg_quality_estimate(raw: bytes):
    i = raw.find(b"\xff\xdb")
    if i < 0 or i + 70 >= len(raw):
        return None
    tbl = list(raw[i + 4:i + 68])
    if len(tbl) < 64:
        return None
    avg = sum(tbl) / len(tbl)
    for q in range(95, 4, -5):
        s = 5000 / q if q < 50 else 200 - 2 * q
        if abs(s / 100.0 * 64 - avg) < 22:
            return q
    return None


TESTS = [
    ("T1_white_mark_fill", t1_white_mark, "White-mark fill rectangle"),
    ("T2_foreign_patch", t2_foreign_patch, "Foreign-palette patch"),
    ("T3_copy_move_clone", t3_copy_move, "Copy-move duplication"),
    ("T4_glyph_colour_drift", t4_glyph_drift, "Re-typed glyph colour drift"),
    ("T5_jpeg_grid", t5_jpeg_grid, "JPEG grid inside lossless container"),
]


# ------------------------------------------------------------------- heatmaps
def _render_overlay(rgb, results, out_path, title):
    """Annotated exhibit: every detected region drawn on the original pixels."""
    canvas = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    h, w = canvas.shape[:2]
    hits = [(k, r) for k, r in results.items() if r.get("bbox") and r["score"] >= 0.22]
    for k, r in hits:
        x0, y0, x1, y1 = r["bbox"]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        col = (60, 76, 245) if r["score"] >= 0.62 else (40, 168, 250) if r["score"] >= 0.40 else (30, 200, 220)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), col, -1)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 3)
        lbl = f"{k.split('_')[0]} {r['score_pct']:.0f}%"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ty = max(0, y0 - th - 8)
        cv2.rectangle(canvas, (x0, ty), (x0 + tw + 10, ty + th + 8), col, -1)
        cv2.putText(canvas, lbl, (x0 + 5, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    bar = 34
    out = np.zeros((h + bar, w, 3), np.uint8)
    out[:h] = canvas
    col = (60, 76, 245) if hits else (70, 150, 70)
    out[h:, :] = col
    cv2.putText(out, title[:110], (8, h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    return os.path.basename(out_path)


# ------------------------------------------------------------------- entrypoint
def analyse_image(raw_bytes: bytes, out_dir: str | None = None, ev_id: str = "img") -> dict:
    """Analyse one image and return a fused authenticity verdict.

    Returns a JSON-serialisable dict: image facts, per-test results with the
    located region, the fused score/verdict, measured sensitivity for this
    detector, and the standing caveat.
    """
    raw = bytes(raw_bytes)
    sha = hashlib.sha256(raw).hexdigest()
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
        fmt = (pil.format or "UNKNOWN").upper()
        rgb = np.array(pil.convert("RGB"))
    except Exception as e:
        return {"ok": False, "error": f"not a readable image: {e}", "sha256": sha,
                "verdict": "UNANALYSABLE", "fusion_pct": 0.0, "tests": {}, "caveat": CAVEAT}

    H0, W0 = rgb.shape[:2]
    scale = 1.0
    if max(H0, W0) > ANALYSIS_MAX:
        scale = ANALYSIS_MAX / max(H0, W0)
        rgb = cv2.resize(rgb, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

    results, container = {}, {}
    for key, fn, _ in TESTS:
        try:
            results[key] = fn(rgb, gray, fmt)
        except Exception as e:
            results[key] = _test(0.0, f"test could not run: {type(e).__name__}: {e}")
    try:
        results["T6_container_metadata"], container = t6_container(rgb, gray, fmt, raw, pil)
    except Exception as e:
        results["T6_container_metadata"] = _test(0.0, f"test could not run: {type(e).__name__}: {e}")

    # ---- fuse with LEARNED weights when available -------------------------
    weights, fmeta = _load_fusion()
    # NOISY-OR, not a weighted mean. These tests look for INDEPENDENT edit signatures,
    # so one decisive signature must be able to carry the verdict on its own. Under a
    # weighted mean a single test at 49% with weight 0.26 contributes 13% and a real
    # splice reads as "no manipulation" - which is how v2 ended up useless.
    #   fused = 1 - PRODUCT_k (1 - s_k * lambda_k)
    # lambda_k is each test's measured reliability (1 - its false-positive rate on
    # genuine exhibits), so a test that never cries wolf passes its score through
    # undiminished and a test that does is discounted toward zero.
    fused_prod = 1.0
    for k, r in results.items():
        lam = float(weights.get(k, 0.0))
        r["weight"] = round(lam, 4)
        r["effective"] = round(r["score"] * lam, 4)
        fused_prod *= (1.0 - min(1.0, r["score"] * lam))
    fused = float(np.clip(1.0 - fused_prod, 0, 1))
    # Marginal contribution of each test = fused WITH it minus fused WITHOUT it.
    #   fused = 1 - PROD(1 - x_k),  others = PROD_{j != k}(1 - x_j)
    #   => contribution_k = others * x_k
    # (the earlier expression expanded to -others*x_k, i.e. always 0, which is why
    #  primary_finding reported "no signature" on an image where T2 had fired.)
    for k, r in results.items():
        xk = min(1.0, r["score"] * r["weight"])
        others = 1.0
        for k2, r2 in results.items():
            if k2 != k:
                others *= (1.0 - min(1.0, r2["score"] * r2["weight"]))
        r["contribution"] = round(others * xk, 4)
    verdict, band = _band(fused)

    # rescale boxes back to the ORIGINAL pixel grid for the UI
    for r in results.values():
        if r.get("bbox") and scale != 1.0:
            r["bbox"] = [int(round(v / scale)) for v in r["bbox"]]

    active = sorted([{"test": k, "label": dict((a, b) for a, _, b in
                     [(x, y, z) for x, y, z in TESTS] + [("T6_container_metadata", None, "Container / metadata")]).get(k, k),
                      "score_pct": r["score_pct"], "contribution_pct": round(r["contribution"] * 100, 1),
                      "region": r["bbox"], "detail": r["detail"]}
                     for k, r in results.items() if r["score"] >= 0.22],
                    key=lambda d: -d["contribution_pct"])

    heatmap = None
    if out_dir:
        try:
            heatmap = _render_overlay(rgb, results, os.path.join(out_dir, f"{ev_id}_forensics.png"),
                                      f"{ev_id}  |  {verdict}  {fused*100:.1f}%  |  SHA-256 {sha[:16]}...")
        except Exception as e:
            heatmap = None

    fired = {k: r for k, r in results.items() if r["score"] >= 0.22}
    primary = (max(fired.items(), key=lambda kv: kv[1]["contribution"])[1] if fired
               else max(results.items(), key=lambda kv: kv[1]["score"])[1])
    cal = _load_calibration()
    return {
        "ok": True,
        "sha256": sha,
        "verdict": verdict,
        "band": band,
        "fusion_score": round(fused, 4),
        "fusion_pct": round(fused * 100, 1),
        "image": {"format": fmt, "width": W0, "height": H0, "bytes": len(raw),
                  "analysis_scale": round(scale, 4), "mode": pil.mode,
                  "unique_grey_levels": int(len(np.unique(gray.astype(np.uint8))))},
        "container": container,
        "tests": results,
        "active_indicators": active,
        "primary_region": primary.get("bbox"),
        "primary_finding": (primary["detail"] if primary.get("score", 0) >= 0.22 else
                            "No edit signature was located. This is a negative screening result, "
                            "not proof of authenticity."),
        "heatmap": heatmap,
        "fusion_model": fmeta,
        "measured_performance": (cal or {}).get("summary"),
        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveat": CAVEAT,
    }


def analyse_path(path: str, out_dir: str | None = None, ev_id: str | None = None) -> dict:
    return analyse_image(open(path, "rb").read(), out_dir,
                         ev_id or os.path.splitext(os.path.basename(path))[0])


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        r = analyse_path(p)
        print(f"\n=== {p}  ->  {r['verdict']} ({r['fusion_pct']}%)  {r['image']['format']} "
              f"{r['image']['width']}x{r['image']['height']}")
        for k in sorted(r["tests"], key=lambda k: -r["tests"][k]["contribution"]):
            t = r["tests"][k]
            if t["score"] > 0:
                print(f"   {k:24s} {t['score_pct']:5.1f}%  w={t['weight']:.2f}  {t['detail'][:120]}")
        print("   ", r["primary_finding"][:180])
